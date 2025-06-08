#!/usr/bin/env python3
"""
llm_forum_scraper_graph.py  —  revision g
=========================================
**New requirement:** In *Step 2* we must:

1. Extract the **page markdown** of the index (with headers/footers/forms/ads
   stripped), **no LLM**.
2. Feed *that markdown* to an **LLM node** that returns every link whose anchor
   text *partially* matches the car keyword (case‑insensitive, any word order).

We therefore update the graph:

```
crawl_index  ──►  llm_find_links  ─► map_threads ─► …
```

* `crawl_index` now stores `index_md` (markdown) instead of raw HTML/anchors.
* `llm_find_links` uses an `LLMExtractionStrategy` with a JSON schema to emit a
  list of `{title, url}` objects matching the keyword **even if only one word
  matches**.
* Removal of the previous BeautifulSoup / res.links logic.

---
Run:
```bash
python llm_forum_scraper_graph.py \
  https://www.ferrarichat.com/forum/tags/for-sale/ "F430 Spider" --debug
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field

# LangGraph ------------------------------------------------------------
from langgraph.graph import Graph, Node, State  # type: ignore

# Crawl4AI -------------------------------------------------------------
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy

###############################################################################
# Config
###############################################################################

MODEL_ID = os.getenv("SCRAPER_MODEL", "ollama/mistral:7b-instruct-q4_0")
EXCLUDED_TAGS = [
    "header",
    "footer",
    "nav",
    "form",
    "aside",
    "script",
    "noscript",
    "style",
]
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
MAX_PARALLEL = int(os.getenv("SCRAPER_MAX_PARALLEL", "6"))

###############################################################################
# Data models
###############################################################################


class SalePost(BaseModel):
    title: str
    price_eur: Optional[float | int]
    location: Optional[str]
    year: Optional[int]
    mileage_km: Optional[int]
    contact: Optional[str]
    images: List[str]
    thread_url: str


class ThreadLink(BaseModel):
    title: str = Field(..., description="Anchor text or title attribute")
    url: str = Field(..., description="Absolute URL to the thread")


class LinkList(BaseModel):
    links: List[ThreadLink]


class LinkState(State):
    index_url: str
    keyword: str
    index_md: Optional[str] = None
    thread_links: List[str] = []
    results: List[Dict[str, Any]] = []


###############################################################################
# Crawl helpers
###############################################################################


async def _crawl_markdown(url: str) -> str:
    cfg = CrawlerRunConfig(
        extraction_strategy=None,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        excluded_tags=EXCLUDED_TAGS,
        user_agent=USER_AGENT,
        stream=False,
    )
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        res = await crawler.arun(url=url, config=cfg)
    if not res.success:
        raise RuntimeError(res.error_message)
    return res.markdown or ""


###############################################################################
# Node implementations
###############################################################################


async def crawl_index(state: LinkState) -> LinkState:
    state.index_md = await _crawl_markdown(state.index_url)
    return state


async def llm_find_links(state: LinkState) -> LinkState:
    if not state.index_md:
        raise ValueError("index_md missing before LLM link extraction")

    prompt = (
        "From the markdown below, return JSON matching the schema. Include every link "
        "where the anchor text, title attribute, OR slug contains **any** of the words "
        f"in the phrase '{state.keyword}' (case‑insensitive). Convert relative URLs to "
        "absolute using the page base. If none, return an empty list."
    )

    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=MODEL_ID),
        extraction_type="schema",
        schema=LinkList.model_json_schema(),
        instruction=prompt,
        input_format="markdown",
        chunk_token_threshold=600,
        overlap_rate=0.1,
        extra_args={"temperature": 0.0, "max_tokens": 1024},
        verbose=False,
    )

    data_uri = "data:text/markdown," + state.index_md
    cfg = CrawlerRunConfig(extraction_strategy=strategy, stream=False)
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        res = await crawler.arun(url=data_uri, config=cfg)
    if not res.success:
        raise RuntimeError(res.error_message)

    try:
        raw = json.loads(res.extracted_content)
        base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(state.index_url))
        links = [ThreadLink(**d) for d in raw["links"]]
        norm = []
        for l in links:
            abs_url = l.url if l.url.startswith("http") else urljoin(base, l.url)
            norm.append(abs_url)
        state.thread_links = list(dict.fromkeys(norm))
    except Exception as e:
        raise ValueError(f"Link JSON parse failed: {e}\n{res.extracted_content}")

    return state


async def crawl_thread(state: LinkState, thread_url: str) -> Dict[str, Any]:
    md = await _crawl_markdown(thread_url)
    prompt = (
        "You are an expert data‑extraction agent. Return ONLY the JSON object with keys: "
        '{"title": str, "price_eur": int|float|null, "location": str|null, '
        '"year": int|null, "mileage_km": int|null, "contact": str|null, "images": [str]}\n'
        "Never wrap output in markdown. End output after the closing }."
    )
    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=MODEL_ID),
        extraction_type="block",
        instruction=prompt,
        input_format="markdown",
        chunk_token_threshold=600,
        overlap_rate=0.1,
        extra_args={"temperature": 0.0, "max_tokens": 1024},
        verbose=False,
    )
    data_uri = "data:text/markdown," + md
    cfg = CrawlerRunConfig(extraction_strategy=strategy, stream=False)
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        res = await crawler.arun(url=data_uri, config=cfg)
    if not res.success:
        raise RuntimeError(res.error_message)
    data = json.loads(res.extracted_content)
    data["thread_url"] = thread_url
    return SalePost(**data).dict()


###############################################################################
# Build LangGraph
###############################################################################

graph = Graph(LinkState)

graph.add_node("crawl_index", Node(crawl_index))
graph.add_node("llm_find_links", Node(llm_find_links))


async def thread_map(state: LinkState):
    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def _runner(url):
        async with sem:
            try:
                return await crawl_thread(state, url)
            except Exception as e:
                return {"error": str(e), "thread_url": url}

    tasks = [_runner(u) for u in state.thread_links]
    for res in await asyncio.gather(*tasks):
        state.results.append(res)
    return state


graph.add_node("map_threads", Node(thread_map))

graph.set_entry("crawl_index")
graph.add_edge("crawl_index", "llm_find_links")


def decide_next(state: LinkState):
    return "map_threads" if state.thread_links else graph.DEFAULT_END


graph.add_conditional_edge("llm_find_links", decide_next)

###############################################################################
# Runner
###############################################################################


async def run_graph(index_url: str, keyword: str, debug: bool = False):
    init = LinkState(index_url=index_url, keyword=keyword)
    final_state = await graph.arun(init)
    if debug:
        print(
            "[DEBUG] Discovered", len(final_state.thread_links), "thread links via LLM"
        )
    print(json.dumps(final_state.results, indent=2, ensure_ascii=False))


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Forum scraper with LangGraph — markdown‑based link discovery via LLM"
    )
    p.add_argument("index_url")
    p.add_argument("car_keyword")
    p.add_argument("--debug", action="store_true")
    cli = p.parse_args()

    asyncio.run(run_graph(cli.index_url, cli.car_keyword, cli.debug))
