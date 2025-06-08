# src/vintage_ai/services/trends_service.py

from __future__ import annotations

from datetime import date
from typing import List
import pandas as pd
from pytrends.request import TrendReq
from vintage_ai.api.core.schemas.v1 import Metrics, TimePoint


def fetch_trends_global(
    query: str, start_date: str = "2006-01-01", granularity: str = "weekly"
) -> pd.DataFrame:
    """
    Fetch global Google Trends interest over time (0-100) for a given query.
    Returns a DataFrame indexed by datetime, with one column 'Global'.
    """
    try:
        end_date = date.today().isoformat()
        timeframe = f"{start_date} {end_date}"
        pytrends = TrendReq(hl="en-US", tz=360)

        # Resolve to a topic ID
        suggestions = pytrends.suggestions(query)
        if not suggestions:
            return pd.DataFrame()

        topic_id = suggestions[0]["mid"]
        pytrends.build_payload([topic_id], timeframe=timeframe, geo="")  # Global

        iot = pytrends.interest_over_time()
        if iot.empty or "isPartial" not in iot.columns:
            return pd.DataFrame()

        series = iot.drop(columns=["isPartial"])[topic_id].rename("Global")

        # Resampling
        if granularity == "monthly":
            return series.resample("M").mean().round(2).to_frame()
        elif granularity == "yearly":
            return series.resample("Y").mean().round(2).to_frame()
        else:  # default: weekly
            return series.to_frame()

    except Exception:
        return pd.DataFrame()


def fetch_trends_metrics(
    car_query: str, start_date: str = "2006-01-01", granularity: str = "weekly"
) -> Metrics:
    """
    Populate only popularity_series in Metrics using fetch_trends_global().
    """
    df = fetch_trends_global(car_query, start_date, granularity)

    # Convert the DataFrame into List[TimePoint]
    series: List[TimePoint] = []
    if not df.empty:
        # df has a single column, which we've named 'Global'
        for ts, row in df.itertuples():
            # when using .itertuples(), row is the 'Global' value
            series.append(TimePoint(timestamp=ts.to_pydatetime(), value=float(row)))

    return Metrics(popularity_series=series or None)
