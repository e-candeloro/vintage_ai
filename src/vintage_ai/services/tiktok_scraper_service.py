# tiktok_sentiment.py
from __future__ import annotations
import os
import json
import logging
import requests
from typing import Any, Dict, List
from collections import Counter
from dotenv import load_dotenv
from statistics import median
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
from vintage_ai.api.core.schemas.v1 import Metrics

load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_API_KEY")
MODEL_PATH = os.getenv(
    "SENTIMENT_MODEL_PATH", "data/models/sentiment_analysis/tabularisai"
)
TIKTOK_RAW_DATA_PATH = os.getenv(
    "TIKTOK_RAW_DATA_PATH", "data/raw/tiktok_search_results.json"
)


SEARCH_ACTOR = "epctex~tiktok-search-scraper"
COMMENT_ACTOR = "clockworks~tiktok-comments-scraper"
SEARCH_URL = f"https://api.apify.com/v2/acts/{SEARCH_ACTOR}/run-sync-get-dataset-items?token={APIFY_TOKEN}"
COMMENT_URL = f"https://api.apify.com/v2/acts/{COMMENT_ACTOR}/run-sync-get-dataset-items?token={APIFY_TOKEN}"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def aggregate_tiktok_sentiment(
    keyword: str, num_videos: int = 10, comments_per_video: int = 20, save_raw=False
) -> dict[str, Any]:
    """
    One-shot helper – search TikTok, grab comments, run sentiment and
    return **median** engagement / sentiment metrics as a plain dict.
    On any fatal error we log and return {} (never crash the caller).
    """

    def _load_pipeline(model_path: str) -> Any:

        try:
            pipe = pipeline(
                "text-classification",
                model="tabularisai/multilingual-sentiment-analysis",
            )

            # Save the model and tokenizer to a local directory
            pipe.save_pretrained(model_path)
            # Load the classification pipeline with the specified model
            pipe = pipeline(
                "text-classification", model=model_path, tokenizer=model_path
            )
        except Exception as e:
            logging.error("Failed to load sentiment pipeline: %s", e)
            raise
        return pipe

    def _post(url: str, payload: dict) -> list[dict]:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    def _search(term: str, limit: int) -> list[dict]:
        return _post(
            SEARCH_URL,
            {"search": [term], "maxItems": limit, "proxy": {"useApifyProxy": True}},
        )

    def _comments(video_url: str, limit: int) -> list[dict]:
        return _post(
            COMMENT_URL,
            {
                "postURLs": [video_url],
                "commentsPerPost": limit,
                "maxRepliesPerComment": 0,
                "resultsPerPage": limit,
            },
        )

    def _clean(item: dict) -> dict:
        s = item.get("stats", {}) or item.get("statsV2", {})

        def _i(x):
            try:
                return int(x)
            except:
                return 0

        return dict(
            likes=_i(s.get("diggCount")),
            shares=_i(s.get("shareCount")),
            plays=_i(s.get("playCount")),
            collections=_i(s.get("collectCount")),
            comments=[
                c.get("text") for c in item.get("comments", []) if isinstance(c, dict)
            ],
        )

    try:
        if not APIFY_TOKEN:
            raise EnvironmentError("APIFY_API_KEY missing")
        videos = [
            v
            for v in tqdm(
                _search(keyword, num_videos), desc="Fetching tiktok videos..."
            )
            if v
        ]
        for v in tqdm(videos[:num_videos], desc="Fetching comments..."):
            url = v.get("url") or v.get("videoUrl") or v.get("shareUrl")
            try:
                v["comments"] = _comments(url, comments_per_video) if url else []
            except Exception as e:
                logging.warning("comment fetch failed for %s: %s", url, e)
                v["comments"] = []

        cleaned = [_clean(v) for v in videos if v.get("comments")]
        if not cleaned:
            raise ValueError("no data")

        if save_raw:
            output_path = Path(TIKTOK_RAW_DATA_PATH)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(videos, f, indent=2, ensure_ascii=False)
            logging.info("Raw search results saved to %s", output_path)

        pipe = _load_pipeline(MODEL_PATH)
        rows = []
        for rec in tqdm(cleaned, desc="Processing comments for sentiment"):
            if not rec["comments"]:
                continue
            preds = pipe(rec["comments"], batch_size=8, truncation=True)
            scores = [p["score"] for p in preds]
            labels = [p["label"].lower() for p in preds]
            total = len(preds)
            total_eng = rec["likes"] + rec["shares"] + rec["collections"] + total
            rows.append(
                dict(
                    num_comments=total,
                    avg_sentiment_score=sum(scores) / total,
                    most_common_sentiment=Counter(labels).most_common(1)[0][0],
                    likes=rec["likes"],
                    shares=rec["shares"],
                    plays=rec["plays"],
                    collections=rec["collections"],
                    engagement_score=(
                        round((total_eng / rec["plays"]) * 100) if rec["plays"] else 0
                    ),
                    overall_sentiment_score=5,  # placeholder
                )
            )

        df = pd.DataFrame(rows)
        med = df.median(numeric_only=True).to_dict()
        med["most_common_sentiment"] = df["most_common_sentiment"].mode().iat[0]
        return Metrics(**med).model_dump()

    except Exception as err:
        logging.error("aggregate_tiktok_sentiment failed: %s", err)
        return {}


if __name__ == "__main__":
    # Example usage
    keyword = "Ferrari Testarossa"
    metrics = aggregate_tiktok_sentiment(
        keyword, num_videos=1, comments_per_video=2, save_raw=True
    )
    print(json.dumps(metrics, indent=2))
