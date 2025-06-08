import os
import time
import random
import json
import logging
from collections import Counter
from typing import List, Optional

import pandas as pd
from instagrapi import Client
from dotenv import load_dotenv
from transformers import pipeline
from tqdm import tqdm

# 👉 Use the official schema from vintage‑ai
from vintage_ai.api.core.schemas.v1 import Metrics

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults (can be overridden via env vars or CLI prompts)
# ---------------------------------------------------------------------------
DEFAULT_N_TOP: int = int(os.getenv("N_TOP", 10))
DEFAULT_N_RECENT: int = int(os.getenv("N_RECENT", 0))
DEFAULT_N_COMMENTS: int = int(os.getenv("N_COMMENTS", 15))
MODEL_PATH: str = os.getenv(
    "SENTIMENT_MODEL_PATH", "data/models/sentiment_analysis/tabularisai"
)


class InstagramSentiment:
    """End‑to‑end hashtag sentiment scraper that returns a :class:`Metrics` object."""

    def __init__(
        self,
        username: str,
        password: str,
        model_path: str = MODEL_PATH,
        *,
        n_top: int = DEFAULT_N_TOP,
        n_recent: int = DEFAULT_N_RECENT,
        n_comments: int = DEFAULT_N_COMMENTS,
    ) -> None:
        self.n_top = n_top
        self.n_recent = n_recent
        self.n_comments = n_comments

        # Login -------------------------------------------------------------
        self.client = Client()
        self.client.login(username, password)

        # Load sentiment model --------------------------------------------
        self.pipe = self._load_pipeline(model_path)

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
        )

    # ------------------------------------------------------------------ Helpers
    @staticmethod
    def _random_delay(min_delay: float = 0.1, max_delay: float = 1.0) -> None:
        """Introduce a pseudo‑human pause between requests."""
        time.sleep(random.uniform(min_delay, max_delay))

    @staticmethod
    def _load_pipeline(model_path: str):
        """Load a sentiment‑analysis pipeline, falling back to a local dir if offline."""
        try:
            return pipeline(
                "text-classification",
                model="tabularisai/multilingual-sentiment-analysis",
            )
        except Exception as err:
            logging.warning("Falling back to local model: %s", err)
            return pipeline(
                "text-classification", model=model_path, tokenizer=model_path
            )

    # ------------------------------------------------ Instagram utilities
    def _fetch_comments(self, media_id: int) -> List[str]:
        """Return up to *n_comments* texts for a media item."""
        try:
            return [
                c.text
                for c in self.client.media_comments(media_id, amount=self.n_comments)
            ]
        except Exception as exc:
            logging.warning("Retry comment fetch due to: %s", exc)
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

    def _clean_media(self, media) -> dict:
        """Convert an *instagrapi* Media object to a plain dict."""
        comments: List[str] = self._fetch_comments(media.id)
        if media.caption_text:
            comments.append(media.caption_text)

        def _safe_int(x):
            try:
                return int(x)
            except Exception:
                return 0

        return dict(
            likes=_safe_int(getattr(media, "like_count", 0)),
            shares=_safe_int(getattr(media, "share_count", 0)),
            plays=_safe_int(
                getattr(media, "play_count", getattr(media, "view_count", 0))
            ),
            collections=_safe_int(getattr(media, "save_count", 0)),
            comments=comments,
        )

    def _scrape_hashtag(self, hashtag: str) -> List[dict]:
        """Return a list of cleaned media dictionaries for *hashtag*."""
        tag = hashtag.replace(" ", "").lstrip("#").lower()
        logging.info("Fetching #%s (top=%s, recent=%s)", tag, self.n_top, self.n_recent)

        medias = self.client.hashtag_medias_top(tag, amount=self.n_top)
        if self.n_recent:
            medias += self.client.hashtag_medias_recent(tag, amount=self.n_recent)

        cleaned: List[dict] = []
        for m in medias:
            logging.debug("Processing media %s", m.pk)
            self._random_delay()
            cleaned.append(self._clean_media(m))
        return cleaned

    # -------------------------------------------------------------- Analysis
    def analyse_hashtag(self, hashtag: str, *, save_json: bool = True) -> Metrics:
        """Scrape, analyse, and return a :class:`Metrics` object for *hashtag*."""
        records = self._scrape_hashtag(hashtag)
        if not records:
            return Metrics()

        rows = []
        for rec in tqdm(records, desc="Sentiment"):
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
            logging.info("Metrics saved to %s", out)

        return metrics


# --------------------------------------------------------------------------- CLI entry‑point
if __name__ == "__main__":
    username = os.getenv("IG_USER")
    password = os.getenv("IG_PASS")
    if not username or not password:
        raise SystemExit("Please set IG_USER and IG_PASS environment variables")

    target = input("Hashtag (without #): ").strip()
    n_top = int(input(f"Top posts [{DEFAULT_N_TOP}]: ") or DEFAULT_N_TOP)
    n_recent = int(input(f"Recent posts [{DEFAULT_N_RECENT}]: ") or DEFAULT_N_RECENT)
    n_comments = int(
        input(f"Comments per post [{DEFAULT_N_COMMENTS}]: ") or DEFAULT_N_COMMENTS
    )

    analyser = InstagramSentiment(
        username=username,
        password=password,
        n_top=n_top,
        n_recent=n_recent,
        n_comments=n_comments,
    )
    result = analyser.analyse_hashtag(target)
    print(json.dumps(result.model_dump(), indent=2))
