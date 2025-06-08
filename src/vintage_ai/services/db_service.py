import duckdb
import datetime
import os
from vintage_ai.api.core.schemas.v1 import Metrics

db_path = os.getenv("DUCKDB_PATH", "data.duckdb")
_con: duckdb.DuckDBPyConnection | None = None  # module-global


def _get_con() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None or _con.close:  # reopen if accidentally closed
        _con = duckdb.connect(db_path)
    return _con


def save_platform_metrics(car_id: str, platform: str, m: Metrics) -> None:
    con = _get_con()
    con.execute(
        """
        INSERT OR REPLACE INTO platform_metrics
        VALUES (?, ?, ?, ?)
        """,
        [car_id, platform, datetime.datetime.utcnow(), m.model_dump_json()],
    )
    print(
        f"[DB] Saved metrics for {car_id} on {platform} at {datetime.datetime.utcnow()}"
    )
