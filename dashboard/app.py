import json
import re
import pandas as pd
import httpx
import os
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
from dotenv import load_dotenv

# --- Initial Setup & Configuration ---
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_BASE_PATH = os.getenv("API_BASE_PATH", "/api")
ENDPOINT = f"{BASE_URL}{API_BASE_PATH}/cars/snapshot"

SAFE_CHARS_PATTERN = re.compile(r"[^a-z0-9\s\-']")


# --- Helper Functions ---
def sanitize_car_name(raw: str) -> str:
    """Cleans the raw car name input."""
    cleaned = SAFE_CHARS_PATTERN.sub("", raw.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def make_df(series: list, label: str) -> pd.DataFrame | None:
    """Converts a list of dictionaries into a clean time-series DataFrame."""
    if not series:
        return None
    df = pd.DataFrame(series)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Remove duplicate timestamps, keeping the first occurrence
    df = df.drop_duplicates(subset="timestamp", keep="first")
    return df.set_index("timestamp")[["value"]].rename(columns={"value": label})


@st.cache_data(ttl=3600, show_spinner="Fetching and processing car data...")
def get_processed_data(car_name: str) -> dict:
    """
    Fetches snapshot data from the API and performs initial processing.
    The results of this function are cached.
    """
    # 1. Fetch from API
    res = httpx.post(ENDPOINT, json={"car_name": car_name}, timeout=15)
    res.raise_for_status()
    data = res.json()
    metrics = data.get("metrics", {})

    # 2. Process time-series data
    price_series = metrics.get("price_series") or []
    pop_series = metrics.get("popularity_series") or []
    df_price = make_df(price_series, "Price")
    df_pop = make_df(pop_series, "Popularity")

    return {
        "car_id": data.get("car_id"),
        "metrics": metrics,
        "df_price": df_price,
        "df_pop": df_pop,
    }


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Classic-Car Popularity Dashboard 🚗✨")

car = st.text_input("Car model", "Ferrari Testarossa")

if st.button("Analyze", type="primary"):
    safe_car = sanitize_car_name(car)
    if not safe_car:
        st.warning("Please enter a valid car name.")
        st.stop()

    try:
        processed_data = get_processed_data(safe_car)
        st.success(f"Successfully loaded metrics for **{processed_data['car_id']}**")
    except httpx.HTTPStatusError as e:
        st.error(
            f"API Error: {e.response.status_code} - {e.response.json().get('detail', 'No details provided.')}"
        )
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

    # Unpack processed data for easier access
    m = processed_data["metrics"]
    df_price = processed_data["df_price"]
    df_pop = processed_data["df_pop"]

    # --- Create Tabs for Dashboard Layout ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Time Series Analysis",
            "🧭 Sentiment Analysis",
            "🔢 Summary Metrics",
            "📄 Raw Data",
        ]
    )

    # ─────────────────────────────────────────────
    # TAB 1: Time Series with dual axes + correlation
    # ─────────────────────────────────────────────
    with tab1:
        st.header("Price and Popularity Over Time")
        if df_price is not None or df_pop is not None:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            if df_price is not None:
                ax1.plot(
                    df_price.index,
                    df_price["Price"],
                    color="tab:blue",
                    label="Price",
                    marker=".",
                )
                ax1.set_ylabel("Price", color="tab:blue")
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

            if df_pop is not None:
                ax2 = ax1.twinx()
                ax2.plot(
                    df_pop.index,
                    df_pop["Popularity"],
                    color="tab:red",
                    label="Popularity",
                    marker=".",
                )
                ax2.set_ylabel("Popularity", color="tab:red")
                ax2.tick_params(axis="y", labelcolor="tab:red")

            ax1.set_xlabel("Date")
            fig.tight_layout()
            st.pyplot(fig)

            # Compute correlation on normalized series
            st.subheader("Analysis & Trends")
            col1, col2 = st.columns(2)
            with col1:
                if df_price is not None and df_pop is not None:
                    df_combined = pd.concat([df_price, df_pop], axis=1).dropna()
                    if not df_combined.empty:
                        p = df_combined["Price"]
                        p_norm = (
                            (p - p.min()) / (p.max() - p.min())
                            if (p.max() - p.min()) > 0
                            else p
                        )
                        pop_norm = df_combined["Popularity"] / 100.0
                        corr = p_norm.corr(pop_norm)
                        st.metric("🔗 Correlation (Price vs Popularity)", f"{corr:.2f}")
                    else:
                        st.info("Not enough overlapping data to compute correlation.")

            with col2:
                if df_pop is not None and not df_pop.empty:
                    cutoff = df_pop.index.max() - pd.DateOffset(years=3)
                    last3 = df_pop[df_pop.index >= cutoff]
                    if not last3.empty:
                        avg_pop = last3["Popularity"].mean() / 100.0
                        st.metric("📊 Last 3 Years Avg Popularity", f"{avg_pop:.2%}")
                    else:
                        st.info("No popularity data in the last 3 years.")
        else:
            st.info("No time series data available for this model.")

    # ─────────────────────────────────────────────
    # TAB 2: Sentiment analysis
    # ─────────────────────────────────────────────
    with tab2:
        st.header("Sentiment Analysis")
        col1, col2 = st.columns(2)
        with col1:
            topics = m.get("topics") or []
            sentiments = m.get("topic_sentiments") or []
            st.subheader("🧭 Top Topics Sentiment")
            if topics and sentiments:
                df_topics = pd.DataFrame(
                    [
                        {"topic": ts["topic"], "sentiment": ts["sentiment"] or 0}
                        for ts in sentiments
                    ]
                ).head(10)
                st.bar_chart(df_topics.set_index("topic")["sentiment"])
            else:
                st.info("No topic sentiment data available.")

        with col2:
            score = m.get("overall_sentiment_score")
            st.subheader("🧠 Overall Sentiment Score")
            if score is not None:
                # Simple gauge using a progress bar
                st.progress(min(max(score, 0.0), 1.0), text=f"Score: {score:.2f}")
            else:
                st.info("No overall sentiment score available.")

    # ─────────────────────────────────────────────
    # TAB 3: Scalar metrics display
    # ─────────────────────────────────────────────
    with tab3:
        st.header("Summary Metrics")
        scalar_keys = [
            "num_comments",
            "likes",
            "shares",
            "plays",
            "collections",
            "engagement_score",
            "avg_sentiment_score",
        ]
        scalars = {
            k.replace("_", " ").title(): m[k]
            for k in scalar_keys
            if m.get(k) is not None
        }

        if scalars:
            cols = st.columns(len(scalars))
            for c, (k, v) in zip(cols, scalars.items()):
                c.metric(k, f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")
        else:
            st.info("No summary metrics available.")

    # ─────────────────────────────────────────────
    # TAB 4: Raw JSON & Table
    # ─────────────────────────────────────────────
    with tab4:
        st.header("Raw Data Explorer")
        st.subheader("Raw Metrics JSON")
        st.json(m)

        st.subheader("Metrics Table")
        df_table = pd.DataFrame(
            [
                {"metric": k, "value": v}
                for k, v in m.items()
                if not isinstance(v, (list, dict))
            ]
        )
        st.dataframe(df_table, use_container_width=True)
