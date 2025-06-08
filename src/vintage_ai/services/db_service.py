import os
import duckdb
import json
import datetime
from vintage_ai.api.core.schemas.v1 import Metrics  # your Pydantic class above

db_path = os.getenv("DUCKDB_PATH", "data.duckdb")  # default fallback


def save_platform_metrics(car_id: str, platform: str, m: Metrics):
    con = duckdb.connect(db_path)
    con.execute(
        """
        INSERT OR REPLACE INTO platform_metrics
        VALUES (?, ?, ?, ?)
        """,
        [car_id, platform, datetime.datetime.utcnow(), m.model_dump_json()],
    )
