import duckdb
import os
from dotenv import load_dotenv
import duckdb

load_dotenv()  # loads .env into environment

db_path = os.getenv("DUCKDB_PATH", "data.duckdb")  # default fallback

con = duckdb.connect(db_path)  # this creates/opens the file

# Create a table if it doesn't exist

con.execute(
    """
CREATE TABLE IF NOT EXISTS platform_metrics (
    car_id       TEXT,
    platform     TEXT,
    run_ts       TIMESTAMP,
    metrics      JSON,
    PRIMARY KEY (car_id, platform, run_ts)
);

CREATE TABLE IF NOT EXISTS overall_cache (
    car_id       TEXT,
    run_ts       TIMESTAMP,
    metrics      JSON,
    PRIMARY KEY (car_id, run_ts)
);
"""
)
