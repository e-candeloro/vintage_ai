import json
import duckdb
import os
from datetime import datetime
from vintage_ai.api.core.schemas.v1 import Metrics, CarSnapshot
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict

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


def aggregate_metrics_and_cache(car_id: str) -> CarSnapshot:

    with duckdb.connect(db_path) as con:
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

        # Step 1: load into Metrics objects
        metrics_objs: list[Metrics] = [
            Metrics(**json.loads(metrics_json)) for _platform, metrics_json in rows
        ]

        # Step 2: aggregate scalar fields by median
        agg_data = defaultdict(list)
        for m in metrics_objs:
            for field in AGGREGATABLE_FIELDS:
                value = getattr(m, field)
                if value is not None:
                    agg_data[field].append(value)

        agg_metrics = {
            field: float(np.median(vals)) for field, vals in agg_data.items()
        }

        # Step 3: enrich with latest price/popularity
        price_rows = con.execute(
            """
            SELECT metric, value
            FROM car_price_popularity
            WHERE car_id = ?
            AND ts >= NOW() - INTERVAL 1 YEAR
        """,
            [car_id],
        ).fetchall()

        latest = {}
        for metric, value in price_rows:
            latest[f"latest_{metric}"] = value

        # Step 4: Compose final Metrics object (non-aggregatable lists excluded for now)
        final_metrics = Metrics(**agg_metrics, **latest)

        # Step 5: Cache to overall_cache
        con.execute(
            """
            INSERT OR REPLACE INTO overall_cache
            VALUES (?, ?, ?)
        """,
            [car_id, datetime.utcnow(), final_metrics.model_dump_json()],
        )

        # Step 6: Return as CarSnapshot
        return CarSnapshot(
            car_id=car_id,
            metrics=final_metrics,
            history=[],  # could populate with earlier snapshots later
        )
