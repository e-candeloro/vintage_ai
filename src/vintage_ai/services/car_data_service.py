"""
Single-responsibility service that

1.  reads the (cached) CSV with prices + popularity
2.  calculates today’s metrics for a requested car
3.  **optionally** calls a user-supplied callback when the car is missing
    (think “trigger a scraper” in phase-2 of the hackathon)
4.  returns a typed `CarSnapshot` object that the FastAPI router can
    send back straight away.

Nothing in this file knows about FastAPI, Streamlit, or scraping details.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr

from vintage_ai.api.core.schemas.v1 import CarMetric, CarSnapshot

load_dotenv()

# ----------------------------------------------------------------------
# Configuration --------------------------------------------------------
# ----------------------------------------------------------------------

DATASET_PATH = Path(
    os.getenv(
        "DATASET_PATH",
        "data/processed/asset_classic_car_prices_with_popularity.csv",
    )
)

# In phase-2 you might pull these from settings / DI container instead.
MissingCarCallback = Callable[[str], None]  # (car_name) -> None

# ----------------------------------------------------------------------
# Internal helpers (never imported elsewhere) --------------------------
# ----------------------------------------------------------------------

_df_cache: Optional[pd.DataFrame] = None


def _get_df() -> pd.DataFrame:
    """Read the CSV once per process – cheap and easy for a hackathon."""
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATASET_PATH)
    return _df_cache


def _popularity_metrics(df: pd.DataFrame, car_name: str) -> Mapping[str, float | None]:
    """
    Pure maths!  Returns a dict of metric_name -> value (or None on failure)
    """
    price_col = car_name.lower()
    pop_col = f"{car_name}_popularity"
    cols = df.columns[1:].to_list()  # skip 'year' column

    if price_col not in cols or pop_col not in cols:
        return {}

    sub_df = df[["year", price_col, pop_col]].dropna()
    if len(sub_df) < 3:
        return {}

    # Stats
    price_z = (sub_df[price_col] - sub_df[price_col].mean()) / sub_df[price_col].std()
    pop_z = (sub_df[pop_col] - sub_df[pop_col].mean()) / sub_df[pop_col].std()
    r, p = pearsonr(price_z, pop_z)

    momentum = sub_df[pop_col].diff().rolling(3).mean().iloc[-1]
    if np.isnan(momentum):
        return {}

    significance_boost = max(0, (1 - min(p, 0.05) / 0.05))
    predictive = max(0, min(100, momentum * significance_boost * 5))

    return {
        "current_popularity": int(sub_df[pop_col].iloc[-1]),
        "correlation_score": round(r * 100, 1),
        "p_value": round(p, 4),
        "predictive_score": round(predictive, 1),
    }


# ----------------------------------------------------------------------
# Public API (imported by FastAPI router) ------------------------------
# ----------------------------------------------------------------------
def load_snapshot(
    car_name: str,
    *,
    on_missing: MissingCarCallback | None = None,
) -> CarSnapshot:
    """
    Compute a `CarSnapshot` for *car_name*.
    - Sanitizes input to lowercase and trims extra spaces.
    - Returns metrics or default values + optional callback on missing car.
    """
    # 🧼 Normalize input to align with sanitized column names
    car_name = car_name.strip().lower()

    df = _get_df()
    metric_dict = _popularity_metrics(df, car_name)

    if not metric_dict:
        if on_missing is not None:
            try:
                on_missing(car_name)
            except Exception:
                pass  # do not break the app due to scraping logic

        # return a result with empty values
        metric_dict = {
            "current_popularity": None,
            "correlation_score": None,
            "p_value": None,
            "predictive_score": None,
        }

    metrics = [CarMetric(metric=k, value=v) for k, v in metric_dict.items()]
    return CarSnapshot(car_name=car_name, as_of=date.today(), metrics=metrics)
