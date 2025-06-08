# src/vintage_ai/services/car_data_service.py
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import List
from collections import defaultdict

import numpy as np
import duckdb

from vintage_ai.api.core.schemas.v1 import Metrics, CarSnapshot, TimePoint
from vintage_ai.services.trends_services import fetch_trends_global
from dotenv import load_dotenv

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
    # Step 1: Read and aggregate platform metrics
    with duckdb.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT platform, metrics
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY platform ORDER BY run_ts DESC) AS rn
                FROM platform_metrics
                WHERE car_id = ?
            )
            WHERE rn = 1
            """,
            [car_id],
        ).fetchall()

        if not rows:
            return CarSnapshot(car_id=car_id, metrics=Metrics(), history=[])

        # Load into Metrics objects
        metrics_objs: List[Metrics] = [Metrics(**json.loads(mj)) for _p, mj in rows]

        # Aggregate scalar fields by median
        agg_data = defaultdict(list)
        for m in metrics_objs:
            for field in AGGREGATABLE_FIELDS:
                value = getattr(m, field)
                if value is not None:
                    agg_data[field].append(value)

        agg_metrics = {
            field: float(np.median(vals)) for field, vals in agg_data.items()
        }

        # Build price and stored popularity series
        price_rows = con.execute(
            """
            SELECT ts, metric, value
            FROM car_price_popularity
            WHERE car_id = ?
            ORDER BY ts ASC
            """,
            [car_id],
        ).fetchall()

        price_series: List[TimePoint] = []
        popularity_series: List[TimePoint] = []
        latest_values: dict[str, float] = {}

        for ts, metric, value in price_rows:
            point = TimePoint(timestamp=ts, value=value)
            if metric == "price":
                price_series.append(point)
                latest_values["latest_price"] = value
            elif metric == "popularity":
                popularity_series.append(point)
                latest_values["latest_popularity"] = value

    # Step 2: Fetch global popularity via Pytrends
    try:
        df_trends = fetch_trends_global(
            car_id, start_date="2006-01-01", granularity="yearly"
        )
        if not df_trends.empty and "Global" in df_trends.columns:
            popularity_series_trends = [
                TimePoint(timestamp=ts.to_pydatetime(), value=float(v))
                for ts, v in df_trends["Global"].items()
            ]
        else:
            popularity_series_trends = []
    except Exception as err:
        logging.warning("Trends pull failed for %s: %s", car_id, err)
        popularity_series_trends = []

    # Step 3: Compose final Metrics (no duplicate keywords)
    final_metrics = Metrics(
        **agg_metrics,
        **latest_values,
        price_series=price_series or None,
        popularity_series=popularity_series_trends or popularity_series or None,
    )

    # Step 4: Cache to overall_cache
    with duckdb.connect(db_path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO overall_cache (car_id, run_ts, metrics)
            VALUES (?, ?, ?)
            """,
            [car_id, datetime.utcnow(), final_metrics.model_dump_json()],
        )

    # Step 5: Return the snapshot
    return CarSnapshot(car_id=car_id, metrics=final_metrics, history=[])


def has_metrics_for_car(car_id: str) -> bool:
    with duckdb.connect(db_path, read_only=True) as con:
        count = con.execute(
            "SELECT COUNT(*) FROM platform_metrics WHERE car_id = ?", [car_id]
        ).fetchone()[0]
    return count > 0
