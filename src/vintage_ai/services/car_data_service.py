# src/vintage_ai/services/car_data_service.py
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
import duckdb

from vintage_ai.api.core.schemas.v1 import (
    Metrics,
    CarSnapshot,
    TimePoint,
)

load_dotenv()

db_path = os.getenv("DUCKDB_PATH", "data.duckdb")

AGGREGATABLE_FIELDS = {
    "num_comments",
    "avg_sentiment_score",
    "likes",
    "shares",
    "plays",
    "collections",
    "engagement_score",
    "overall_sentiment_score",
}


def aggregate_snapshot(car_id: str) -> CarSnapshot:
    with duckdb.connect(db_path) as con:
        # -- Step 1: latest platform metrics
        rows = con.execute(
            """
            SELECT platform, metrics
            FROM (
                SELECT *, row_number() OVER (PARTITION BY platform ORDER BY run_ts DESC) AS rn
                FROM platform_metrics
                WHERE car_id = ?
            )
            WHERE rn = 1
        """,
            [car_id],
        ).fetchall()

        if not rows:
            return CarSnapshot(car_id=car_id, metrics=Metrics(), history=[])

        # -- Step 2: load and aggregate scalar fields
        metrics_objs = [
            Metrics(**json.loads(metrics_json)) for _p, metrics_json in rows
        ]

        agg_data = defaultdict(list)
        for m in metrics_objs:
            for field in AGGREGATABLE_FIELDS:
                value = getattr(m, field)
                if value is not None:
                    agg_data[field].append(value)

        agg_metrics = {
            field: float(np.median(vals)) for field, vals in agg_data.items()
        }

        # -- Step 3: build price/popularity time series
        series_rows = con.execute(
            """
            SELECT ts, metric, value
            FROM car_price_popularity
            WHERE car_id = ?
              AND ts >= NOW() - INTERVAL 1 YEAR
            ORDER BY ts ASC
        """,
            [car_id],
        ).fetchall()

        price_series = []
        popularity_series = []
        latest_values = {}

        for ts, metric, value in series_rows:
            point = TimePoint(timestamp=ts, value=value)
            if metric == "price":
                price_series.append(point)
                latest_values["latest_price"] = value
            elif metric == "popularity":
                popularity_series.append(point)
                latest_values["latest_popularity"] = value

        # -- Step 4: compose final Metrics object
        final_metrics = Metrics(
            **agg_metrics,
            **latest_values,
            price_series=price_series or None,
            popularity_series=popularity_series or None,
        )

        # -- Step 5: cache the result
        con.execute(
            """
            INSERT OR REPLACE INTO overall_cache
            VALUES (?, ?, ?)
        """,
            [car_id, datetime.utcnow(), final_metrics.model_dump_json()],
        )

        # -- Step 6: return result
        return CarSnapshot(
            car_id=car_id,
            metrics=final_metrics,
            history=[],
        )


def has_metrics_for_car(car_id: str) -> bool:
    # 🔐 Safe: one short-lived connection
    with duckdb.connect(db_path, read_only=True) as con:
        count = con.execute(
            """
            SELECT COUNT(*) FROM platform_metrics WHERE car_id = ?
        """,
            [car_id],
        ).fetchone()[0]
    return count > 0
