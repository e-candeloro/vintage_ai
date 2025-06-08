# src/vintage_ai/core/schemas/v1.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pandas as pd
from typing import List, Optional, Sequence, Literal
from datetime import date, datetime
from datetime import date
from pydantic import Field, field_validator
from .base import BaseSchema

# --- request ---


class CarQuery(BaseSchema):
    car_name: str = Field(..., max_length=60, examples=["Porsche 911"])

    @field_validator("car_name")
    @classmethod
    def clean_car_name(cls, v: str) -> str:
        return v.strip().lower()


# --- response (flexible wrapper) ---


# ── leaf helpers ──────────────────────────────────────────────────────


class TimePoint(BaseModel):
    timestamp: datetime
    value: float | int | None = None  # price, popularity … anything


class TopicSentiment(BaseModel):
    topic: str
    sentiment: float | None = None  # e.g. −1 … +1


# ── the universal blob sent by every scraper ─────────────────────────


class Metrics(BaseModel):
    # platform-agnostic scalar KPIs (all optional)
    num_comments: Optional[int] = None
    avg_sentiment_score: Optional[float] = None
    most_common_sentiment: Optional[str] = None
    likes: Optional[int] = None
    shares: Optional[int] = None
    plays: Optional[int] = None
    collections: Optional[int] = None
    engagement_score: Optional[float] = None
    overall_sentiment_score: Optional[float] = None

    # extra cross-platform goodies
    topics: Optional[List[str]] = None
    # len == topics
    topic_sentiments: Optional[List[TopicSentiment]] = None
    price_series: Optional[List[TimePoint]] = None
    popularity_series: Optional[List[TimePoint]] = None

    # future proof: silently accept new keys
    model_config = ConfigDict(extra="allow")

    # keep lengths aligned
    @field_validator("topic_sentiments")
    @classmethod
    def _same_len(cls, v, info):
        topics = info.data.get("topics")
        if topics and v and len(topics) != len(v):
            raise ValueError("topic_sentiments must match length of topics")
        return v


class CarSnapshot(BaseModel):
    car_id: str
    metrics: Metrics  # latest snapshot
    history: List[dict]  # List of historical snapshots if available


class CarPending(BaseModel):
    status: Literal["pending"]
    task_id: str
    car_name: str


# class ResponseCarMetric(BaseSchema):
#     metric: str  # e.g. "current_popularity"
#     value: Optional[Metrics]  # None when the calc fails


# class CarSnapshot(BaseSchema):
#     car_name: str
#     as_of: date
#     metrics: List[ResponseCarMetric]  # **flexible** – add/drop metrics at will
