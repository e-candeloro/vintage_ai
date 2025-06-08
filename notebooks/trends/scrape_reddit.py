import asyncio
import json
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

HEADERS = {
    # Spoof a real browser UA to dodge basic bot‑blocking
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}

BASE = "https://www.reddit.com"
JSON_SEARCH = (
    "{base}/search.json?q={query}&limit=100&sort=relevance&raw_json=1&after={after}"
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def smart_chunk(text: str, max_len: int = 1000) -> List[str]:
    """Simple header‑aware chunker so every chunk <= max_len characters."""

    def split(hpat: str, md: str):
        idx = [m.start() for m in re.finditer(hpat, md, re.M)]
        idx.append(len(md))
        return [
            md[idx[i] : idx[i + 1]].strip()
            for i in range(len(idx) - 1)
            if md[idx[i] : idx[i + 1]].strip()
        ]

    chunks: List[str] = []
    for h1 in split(r"^# ", text):
        if len(h1) > max_len:
            for h2 in split(r"^## ", h1):
                chunks += [
                    h2[i : i + max_len].strip() for i in range(0, len(h2), max_len)
                ]
        else:
            chunks.append(h1)
    out: List[str] = []
    for c in chunks:
        out += [c[i : i + max_len].strip() for i in range(0, len(c), max_len)]
    return [c for c in out if c]


async def _get_json(
    crawler, url: str, retries: int = 3, pause: tuple[int, int] = (2, 5)
) -> dict:
    """Fetch a JSON url with basic 429 back‑off and validation."""
    for attempt in range(retries + 1):
        res = await crawler.arun(url, headers=HEADERS)
        code = res.status_code
        if code == 200 and res.html.strip().startswith("{"):
            try:
                return json.loads(res.html)
            except json.JSONDecodeError:
                return {}
        if code == 429 and attempt < retries:
            wait = random.randint(*pause)
            print(f"[429] sleeping {wait}s …")
            time.sleep(wait)
            continue
        break
    return {}


async def collect_post_urls(query: str, pages: int, crawler) -> List[str]:
    """Use the modern Reddit JSON search endpoint; fallback to HTML if necessary."""
    urls: List[str] = []
    after = ""
    for i in range(pages):
        search_url = JSON_SEARCH.format(base=BASE, query=quote_plus(query), after=after)
        data = await _get_json(crawler, search_url)
        if not data:
            print("[WARN] JSON search blocked. Falling back to HTML search page…")
            urls += await collect_post_urls_html(query, pages - i, crawler)
            break

        children = data.get("data", {}).get("children", [])
        for post in children:
            if post.get("kind") == "t3":
                permalink = post["data"].get("permalink")
                if permalink:
                    urls.append(f"{BASE}{permalink}.json")
        after = data.get("data", {}).get("after") or ""
        print(
            f"[INFO] JSON page {i+1}: collected {len(children)} posts (after={after})."
        )
        if not after:
            break
    return list(dict.fromkeys(urls))  # dedupe while preserving order


async def collect_post_urls_html(query: str, pages: int, crawler) -> List[str]:
    """Fallback HTML scraper (old implementation)."""
    urls: set[str] = set()
    for i in range(pages):
        html_url = f"{BASE}/search?q={quote_plus(query)}"
        res = await crawler.arun(html_url, headers=HEADERS)
        if not res.success:
            break
        soup = BeautifulSoup(res.html, "html.parser")
        new_links = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/r/") and "/comments/" in href:
                full = f"{BASE}{href.split('?')[0].rstrip('/')}.json"
                if full not in urls:
                    urls.add(full)
                    new_links += 1
        print(f"[INFO] HTML page {i+1}: +{new_links} new links.")
        if new_links == 0:
            break
    return list(urls)


async def scrape_reddit_query(
    query: str,
    pages: int = 3,
    max_comments: int = 300,
    max_chunksize: int = 1000,
    concurrency: int = 8,
) -> Tuple[List[str], List[Dict]]:
    """Main entrypoint: returns (documents, metadatas)."""
    dispatcher = MemoryAdaptiveDispatcher(max_session_permit=concurrency)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    browser = BrowserConfig(headless=True, verbose=False)

    documents: List[str] = []
    metas: List[Dict] = []
    chunk_idx = 0

    async with AsyncWebCrawler(config=browser) as crawler:
        post_urls = await collect_post_urls(query, pages, crawler)
        if not post_urls:
            print("[WARN] No posts found.")
            return [], []
        print(f"[INFO] Total post URLs: {len(post_urls)}")

        results = await crawler.arun_many(
            post_urls, dispatcher=dispatcher, config=run_cfg
        )
        print(
            f"[INFO] Retrieved {sum(r.success for r in results)} / {len(results)} post JSONs."
        )

        for res in results:
            if not res.success or not res.html.strip().startswith("["):
                continue
            try:
                j = json.loads(res.html)
            except json.JSONDecodeError:
                continue
            if len(j) < 2:
                continue
            post = j[0]["data"]["children"][0]["data"]
            title = post.get("title", "(no‑title)")
            body_md = f"# {title}\n\n{post.get('selftext', '')}"

            for com in j[1]["data"].get("children", []):
                if com.get("kind") != "t1":
                    continue
                cd = com.get("data", {})
                body_md += (
                    f"\n\n## Comment by {cd.get('author')}\n\n{cd.get('body', '')}"
                )

            for chunk in smart_chunk(body_md, max_len=max_chunksize):
                documents.append(chunk)
                metas.append(
                    {
                        "chunk_index": chunk_idx,
                        "source": f"{BASE}{post.get('permalink', '')}",
                        "title": title,
                        "created_utc": datetime.utcfromtimestamp(
                            post.get("created_utc", 0)
                        ).isoformat(),
                        "char_count": len(chunk),
                        "word_count": len(chunk.split()),
                        "headers": "; ".join(re.findall(r"^(#.+)$", chunk, re.M)),
                        "query": query,
                    }
                )
                chunk_idx += 1

    return documents, metas


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Search phrase, e.g. 'Ferrari Testarossa'")
    ap.add_argument("--pages", type=int, default=3)
    args = ap.parse_args()

    docs, meta = asyncio.run(scrape_reddit_query(args.query, pages=args.pages))
    print(f"Scraped {len(docs)} chunks for query '{args.query}'.")
    if meta:
        print(json.dumps(meta[0], indent=2))
