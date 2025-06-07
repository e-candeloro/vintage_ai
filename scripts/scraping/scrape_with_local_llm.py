#!/usr/bin/env python3
"""
llm_forum_scraper_graph.py  —  revision j (LangGraph API fix: `state_schema`)
==========================================================================
* Removes the `State` import and inheritance (no longer exported in new
  LangGraph). Instead, we **pass `state_schema=LinkState` when creating the
  graph**. `LinkState` is now a `TypedDict`, which is the recommended simple
  schema.
* No functional changes to the nodes or prompts.

Install/Upgrade
---------------
```bash
pip install -U langgraph
```

Run
----
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
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field
from langgraph.graph import Graph  # State removed
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy

###############################################################################
# Config constants
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
    title: str
    url: str


class LinkList(BaseModel):
    links: List[ThreadLink]


class LinkState(TypedDict, total=False):
    index_url: str
    keyword: str
    index_md: Optional[str]
    thread_links: List[str]
    results: List[Dict[str, Any]]
    debug: bool


###############################################################################
# Crawl helper
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
# Graph nodes
###############################################################################


async def crawl_index(state: LinkState) -> LinkState:
    state["index_md"] = await _crawl_markdown(state["index_url"])
    if state.get("debug"):
        print("[DEBUG] First 500 chars of index markdown:\n", state["index_md"][:500])
    return state


async def llm_find_links(state: LinkState) -> LinkState:
    keyword = state["keyword"]
    example_json = {
        "links": [
            {"title": "F430 Spider – Rosso Corsa", "url": "/threads/f430-spider-12345/"}
        ]
    }
    prompt = (
        "You are an expert link‑extraction agent. From the markdown below, output **only** a JSON object matching the schema.\n\n"
        f"Schema example (one element):\n```json\n{json.dumps(example_json, ensure_ascii=False)}\n```\n\n"
        f'Include every link whose anchor text, title attribute, *or* URL slug contains at least one word from "{keyword}" (case‑insensitive, any order).\n'
        'Convert relative URLs to absolute using the page base. If nothing matches, return {"links": []}.\n'
        "Do NOT wrap the JSON in markdown fences other than the example above."
    )
    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=MODEL_ID),
        extraction_type="schema",
        schema=LinkList.model_json_schema(),
        instruction=prompt,
        input_format="markdown",
        chunk_token_threshold=0,
        apply_chunking=False,
        overlap_rate=0.0,
        extra_args={"temperature": 0.0, "max_tokens": 1024},
        verbose=False,
    )
    data_uri = "data:text/markdown," + state.get("index_md", "")
    cfg = CrawlerRunConfig(extraction_strategy=strategy, stream=False)
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        res = await crawler.arun(url=data_uri, config=cfg)
    if not res.success:
        raise RuntimeError(res.error_message)
    raw_json = res.extracted_content
    if state.get("debug"):
        print("[DEBUG] Raw link JSON from LLM:\n", raw_json)
    try:
        base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(state["index_url"]))
        links = [ThreadLink(**d) for d in json.loads(raw_json)["links"]]
        abs_urls = [
            l.url if l.url.startswith("http") else urljoin(base, l.url) for l in links
        ]
        state["thread_links"] = list(dict.fromkeys(abs_urls))
    except Exception as e:
        if state.get("debug"):
            print("[WARN] Failed to parse link JSON:", e)
        state["thread_links"] = []
    return state


async def crawl_thread(state: LinkState, thread_url: str) -> Dict[str, Any]:
    md = await _crawl_markdown(thread_url)
    prompt = (
        "You are an expert data‑extraction agent. Return ONLY the JSON object with keys: "
        '{"title": str, "price_eur": int|float|null, "location": str|null, "year": int|null, '
        '"mileage_km": int|null, "contact": str|null, "images": [str]}\n'
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


async def map_threads(state: LinkState) -> LinkState:
    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def _run(url):
        async with sem:
            try:
                return await crawl_thread(state, url)
            except Exception as e:
                return {"error": str(e), "thread_url": url}

    tasks = [_run(u) for u in state.get("thread_links", [])]
    for res in await asyncio.gather(*tasks):
        state.setdefault("results", []).append(res)
    return state


###############################################################################
# Build graph
###############################################################################

graph = Graph(state_schema=LinkState)

graph.add_node("crawl_index", crawl_index)

graph.add_node("llm_find_links", llm_find_links)

graph.add_node("map_threads", map_threads)

graph.set_entry("crawl_index")

graph.add_edge("crawl_index", "llm_find_links")

graph.add_conditional_edge(
    "llm_find_links",
    lambda s: "map_threads" if s.get("thread_links") else graph.DEFAULT_END,
)

###############################################################################
# Runner
###############################################################################


async def run_graph(index_url: str, keyword: str, debug: bool = False):
    init: LinkState = {"index_url": index_url, "keyword": keyword, "debug": debug}
    final_state = await graph.arun(init)
