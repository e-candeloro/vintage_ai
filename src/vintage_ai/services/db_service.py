import duckdb
import json
import datetime
from vintage_ai.api.core.schemas.v1 import Metrics  # your Pydantic class above


def save_platform_metrics(car_id: str, platform: str, m: Metrics):
    con = duckdb.connect("data.duckdb")
    con.execute(
        """
        INSERT OR REPLACE INTO platform_metrics
        VALUES (?, ?, ?, ?)
        """,
        [car_id, platform, datetime.datetime.utcnow(), m.model_dump_json()],
    )
