"""sentiment_metrics.py
=======================
Unified sentiment‑scraping utilities for Instagram and TikTok.
Each platform has its own class that logs in / authenticates, fetches data,
runs multilingual sentiment analysis and returns a **Metrics** object
(from `vintage_ai.api.core.schemas.v1`).

Classes exported
----------------
- **InstagramSentiment** – scrape hashtag posts with instagrapi.
- **TikTokSentiment**     – scrape videos & comments via Apify actors.

Both expose a single public method:
    analyse_<platform>()  -> Metrics

Environment variables (all optional unless noted):
- IG_USER / IG_PASS                – Instagram credentials  ❗ required for IG
- APIFY_API_KEY                    – Apify token           ❗ required for TikTok
- SENTIMENT_MODEL_PATH             – fallback local HF model dir
- N_TOP / N_RECENT / N_COMMENTS    – IG defaults override
- NUM_VIDEOS / COMMENTS_PER_VIDEO  – TikTok defaults override
- TIKTOK_RAW_DATA_PATH             – where to persist raw TikTok JSON

The module can be invoked directly (`python sentiment_metrics.py`) for a quick
CLI demo of either platform.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from instagrapi import Client
from transformers import pipeline
from tqdm import tqdm

# 👉 Official schema
from vintage_ai.api.core.schemas.v1 import Metrics

# ---------------------------------------------------------------------------
# ENV & DEFAULTS
# ---------------------------------------------------------------------------
load_dotenv()

# --- Instagram defaults ----------------------------------------------------
DEFAULT_N_TOP: int = int(os.getenv("N_TOP", 10))
DEFAULT_N_RECENT: int = int(os.getenv("N_RECENT", 0))
DEFAULT_N_COMMENTS: int = int(os.getenv("N_COMMENTS", 15))

# --- TikTok defaults -------------------------------------------------------
DEFAULT_NUM_VIDEOS: int = int(os.getenv("NUM_VIDEOS", 10))
DEFAULT_COMMENTS_PER_VIDEO: int = int(os.getenv("COMMENTS_PER_VIDEO", 20))
TIKTOK_RAW_DATA_PATH: str = os.getenv(
    "TIKTOK_RAW_DATA_PATH", "data/raw/tiktok_search_results.json"
)

# --- Shared ---------------------------------------------------------------
MODEL_PATH: str = os.getenv(
    "SENTIMENT_MODEL_PATH", "data/models/sentiment_analysis/tabularisai"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------------------------
# Helper: universal sentiment pipeline loader
# ---------------------------------------------------------------------------


def _load_sentiment_pipeline(model_path: str):
    """Return a HF text‑classification pipeline – online if available, else local."""
    try:
        return pipeline(
            "text-classification", model="tabularisai/multilingual-sentiment-analysis"
        )
    except Exception as err:
        logging.warning("Falling back to local model (%s): %s", model_path, err)
        return pipeline("text-classification", model=model_path, tokenizer=model_path)


# ===========================================================================
# Instagram
# ===========================================================================
class InstagramSentiment:
    """End‑to‑end hashtag scraper returning a :class:`Metrics`."""

    def __init__(
        self,
        username: str,
        password: str,
        *,
        n_top: int = DEFAULT_N_TOP,
        n_recent: int = DEFAULT_N_RECENT,
        n_comments: int = DEFAULT_N_COMMENTS,
        model_path: str = MODEL_PATH,
    ) -> None:
        self.n_top = n_top
        self.n_recent = n_recent
        self.n_comments = n_comments

        # Login ---------------------------------------------------------
        self.client = Client()
        self.client.login(username, password)

        # Sentiment model ----------------------------------------------
        self.pipe = _load_sentiment_pipeline(model_path)

    # ---------------------------------------------------------------- Helpers
    @staticmethod
    def _random_delay(min_delay: float = 0.1, max_delay: float = 1.0) -> None:
        time.sleep(random.uniform(min_delay, max_delay))

    # ------------------------------------------------ Instagram utilities
    def _fetch_comments(self, media_id: int) -> List[str]:
        try:
            return [
                c.text
                for c in self.client.media_comments(media_id, amount=self.n_comments)
            ]
        except Exception as exc:
            logging.warning("Retry comment fetch: %s", exc)
            self._random_delay(1, 2)
            try:
                return [
                    c.text
                    for c in self.client.media_comments(
                        media_id, amount=self.n_comments
                    )
                ]
            except Exception:
                return []

    def _clean_media(self, media):
        cmts: List[str] = self._fetch_comments(media.id)
        if media.caption_text:
            cmts.append(media.caption_text)

        def _i(x):
            try:
                return int(x)
            except Exception:
                return 0

        return dict(
            likes=_i(getattr(media, "like_count", 0)),
            shares=_i(getattr(media, "share_count", 0)),
            plays=_i(getattr(media, "play_count", getattr(media, "view_count", 0))),
            collections=_i(getattr(media, "save_count", 0)),
            comments=cmts,
        )

    def _scrape_hashtag(self, hashtag: str) -> List[dict]:
        tag = hashtag.replace(" ", "").lstrip("#").lower()
        logging.info(
            "Fetching IG #%s (top=%s recent=%s)", tag, self.n_top, self.n_recent
        )

        medias = self.client.hashtag_medias_top(tag, amount=self.n_top)
        if self.n_recent:
            medias += self.client.hashtag_medias_recent(tag, amount=self.n_recent)

        cleaned = []
        for m in medias:
            self._random_delay()
            cleaned.append(self._clean_media(m))
        return cleaned

    # -------------------------------------------------------------- Public API
    def analyse_hashtag(self, hashtag: str, *, save_json: bool = False) -> Metrics:
        records = self._scrape_hashtag(hashtag)
        if not records:
            return Metrics()

        rows = []
        for rec in tqdm(records, desc="IG sentiment"):
            if not rec["comments"]:
                continue
            preds = self.pipe(rec["comments"], batch_size=8, truncation=True)
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
                )
            )

        if not rows:
            return Metrics()

        df = pd.DataFrame(rows)
        metrics_dict = df.median(numeric_only=True).to_dict()
        metrics_dict["most_common_sentiment"] = (
            df["most_common_sentiment"].mode().iat[0]
        )
        metrics_dict["overall_sentiment_score"] = df["avg_sentiment_score"].mean()

        metrics = Metrics(**metrics_dict)

        if save_json:
            out = f"hashtag_{hashtag.lstrip('#').lower()}_metrics.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(metrics.model_dump(), fh, indent=2, ensure_ascii=False)
            logging.info("IG metrics saved to %s", out)

        return metrics


# ===========================================================================
# TikTok
# ===========================================================================
class TikTokSentiment:
    """TikTok search + comment scraper that aggregates sentiment & engagement."""

    SEARCH_ACTOR = "epctex~tiktok-search-scraper"
    COMMENT_ACTOR = "clockworks~tiktok-comments-scraper"

    def __init__(
        self,
        *,
        apify_token: str | None = None,
        num_videos: int = DEFAULT_NUM_VIDEOS,
        comments_per_video: int = DEFAULT_COMMENTS_PER_VIDEO,
        model_path: str = MODEL_PATH,
        raw_save_path: str | Path | None = TIKTOK_RAW_DATA_PATH,
    ) -> None:
        self.token = apify_token or os.getenv("APIFY_API_KEY")
        if not self.token:
            raise EnvironmentError("APIFY_API_KEY missing – cannot scrape TikTok")

        self.num_videos = num_videos
        self.comments_per_video = comments_per_video
        self.raw_save_path = Path(raw_save_path) if raw_save_path else None

        self.SEARCH_URL = f"https://api.apify.com/v2/acts/{self.SEARCH_ACTOR}/run-sync-get-dataset-items?token={self.token}"
        self.COMMENT_URL = f"https://api.apify.com/v2/acts/{self.COMMENT_ACTOR}/run-sync-get-dataset-items?token={self.token}"

        self.pipe = _load_sentiment_pipeline(model_path)

    # ---------------------------------------------------------------- Helpers
    def _post(self, url: str, payload: dict) -> list[dict]:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    def _search(self, term: str) -> list[dict]:
        return self._post(
            self.SEARCH_URL,
            {
                "search": [term],
                "maxItems": self.num_videos,
                "proxy": {"useApifyProxy": True},
            },
        )

    def _comments(self, video_url: str) -> list[dict]:
        return self._post(
            self.COMMENT_URL,
            {
                "postURLs": [video_url],
                "commentsPerPost": self.comments_per_video,
                "maxRepliesPerComment": 0,
                "resultsPerPage": self.comments_per_video,
            },
        )

    @staticmethod
    def _clean(item: dict) -> dict:
        s = item.get("stats", {}) or item.get("statsV2", {})

        def _i(x):
            try:
                return int(x)
            except Exception:
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

    # -------------------------------------------------------------- Public API
    def analyse_keyword(self, keyword: str, *, save_raw: bool = False) -> Metrics:
        try:
            videos = [v for v in self._search(keyword) if v]
            for v in tqdm(videos, desc="TikTok comments"):
                url = v.get("url") or v.get("videoUrl") or v.get("shareUrl")
                try:
                    v["comments"] = self._comments(url) if url else []
                except Exception as e:
                    logging.warning("Comment fetch failed for %s: %s", url, e)
                    v["comments"] = []

            cleaned = [self._clean(v) for v in videos if v.get("comments")]
            if not cleaned:
                return Metrics()

            # Persist raw if needed ----------------------------------
            if save_raw and self.raw_save_path:
                self.raw_save_path.parent.mkdir(parents=True, exist_ok=True)
                with self.raw_save_path.open("w", encoding="utf-8") as f:
                    json.dump(videos, f, indent=2, ensure_ascii=False)
                logging.info("Raw TikTok search saved to %s", self.raw_save_path)

            # Sentiment ---------------------------------------------
            rows = []
            for rec in tqdm(cleaned, desc="TikTok sentiment"):
                if not rec["comments"]:
                    continue
                preds = self.pipe(rec["comments"], batch_size=8, truncation=True)
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
                            round((total_eng / rec["plays"]) * 100)
                            if rec["plays"]
                            else 0
                        ),
                    )
                )

            if not rows:
                return Metrics()

            df = pd.DataFrame(rows)
            metrics_dict = df.median(numeric_only=True).to_dict()
            metrics_dict["most_common_sentiment"] = (
                df["most_common_sentiment"].mode().iat[0]
            )
            metrics_dict["overall_sentiment_score"] = df["avg_sentiment_score"].mean()

            return Metrics(**metrics_dict)

        except Exception as err:
            logging.error("TikTok analyse_keyword failed: %s", err)
            return Metrics()


# ---------------------------------------------------------------------------
# CLI helper (for quick tests)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    platform = input("Platform (ig/tk): ").strip().lower()

    if platform == "ig":
        username = os.getenv("IG_USER")
        password = os.getenv("IG_PASS")
        if not username or not password:
            raise SystemExit("Set IG_USER and IG_PASS env vars")

        tag = input("Hashtag (without #): ").strip()
        analyser = InstagramSentiment(username, password)
        res = analyser.analyse_hashtag(tag, save_json=True)
    elif platform == "tk":
        kw = input("Keyword: ").strip()
        analyser = TikTokSentiment()
        res = analyser.analyse_keyword(kw, save_raw=True)
    else:
        raise SystemExit("Unknown platform – choose 'ig' or 'tk'.")

    print(json.dumps(res.model_dump(), indent=2))
