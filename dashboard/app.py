import json
import re
import pandas as pd
import httpx
import asyncio
import os
import matplotlib.pyplot as plt

import streamlit as st
from datetime import date
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_BASE_PATH = os.getenv("API_BASE_PATH", "/api")
ENDPOINT = f"{BASE_URL}{API_BASE_PATH}/cars/snapshot"

SAFE_CHARS_PATTERN = re.compile(r"[^a-z0-9\s\-']")


def sanitize_car_name(raw: str) -> str:
    cleaned = SAFE_CHARS_PATTERN.sub("", raw.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_snapshot(car_name: str) -> dict:
    res = httpx.post(ENDPOINT, json={"car_name": car_name}, timeout=15)
    res.raise_for_status()
    return res.json()


st.title("Classic-Car Popularity Dashboard 🚗✨")

car = st.text_input("Car model", "Ferrari Testarossa")

if st.button("Analyze"):
    safe_car = sanitize_car_name(car)
    if not safe_car:
        st.warning("Please enter a valid car name.")
        st.stop()

    with st.spinner("Calling backend …"):
        try:
            data = fetch_snapshot(safe_car)
        except httpx.HTTPStatusError as e:
            st.error(f"{e.response.status_code}: {e.response.json().get('detail')}")
            st.stop()

    st.success(f"Metrics for **{data['car_id']}**")
    m = data["metrics"]

    # ─────────────────────────────────────────────
    # 1) Time Series with dual axes
    # ─────────────────────────────────────────────
    price = m.get("price_series") or []
    pop = m.get("popularity_series") or []

    def make_df(series, label):
        if not series:
            return None
        df = pd.DataFrame(series)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")[["value"]].rename(columns={"value": label})

    df_price = make_df(price, "Price")
    df_pop = make_df(pop, "Popularity")

    if df_price is not None or df_pop is not None:
        fig, ax1 = plt.subplots()
        if df_price is not None:
            df_price = df_price[~df_price.index.duplicated()]
            ax1.plot(df_price.index, df_price["Price"], color="tab:blue", label="Price")
            ax1.set_ylabel("Price", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

        if df_pop is not None:
            ax2 = ax1.twinx()
            df_pop = df_pop[~df_pop.index.duplicated()]
            ax2.plot(
                df_pop.index, df_pop["Popularity"], color="tab:red", label="Popularity"
            )
            ax2.set_ylabel("Popularity", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.tight_layout()
        st.subheader("📈 Price & Popularity Over Time")
        st.pyplot(fig)
    else:
        st.info("No time series available.")

    # ─────────────────────────────────────────────
    # 2) Topics sentiment bar chart
    # ─────────────────────────────────────────────
    topics = m.get("topics") or []
    sentiments = m.get("topic_sentiments") or []

    if topics and sentiments:
        df_topics = pd.DataFrame(
            [
                {"topic": ts["topic"], "sentiment": ts["sentiment"] or 0}
                for ts in sentiments
            ]
        )
        df_topics = df_topics.head(10)  # top 10
        st.subheader("🧭 Top Topics Sentiment")
        st.bar_chart(df_topics.set_index("topic")["sentiment"])
    else:
        st.info("No topics sentiment available.")

    # ─────────────────────────────────────────────
    # 3) Overall sentiment score gauge
    # ─────────────────────────────────────────────
    score = m.get("overall_sentiment_score")
    if score is not None:
        st.subheader("🧠 Overall Sentiment Score")
        st.progress(min(max(score, 0.0), 1.0), text=f"{score:.2f}")
    else:
        st.info("No overall sentiment score.")

    # ─────────────────────────────────────────────
    # 4) Scalar metrics display
    # ─────────────────────────────────────────────
    scalar_keys = [
        "num_comments",
        "likes",
        "shares",
        "plays",
        "collections",
        "engagement_score",
        "avg_sentiment_score",
    ]
    scalars = {k: m[k] for k in scalar_keys if m.get(k) is not None}

    if scalars:
        st.subheader("🔢 Summary Metrics")
        cols = st.columns(len(scalars))
        for c, (k, v) in zip(cols, scalars.items()):
            c.metric(
                k.replace("_", " ").title(), f"{v:,.2f}" if isinstance(v, float) else v
            )

    # ─────────────────────────────────────────────
    # 5) Raw JSON & Table
    # ─────────────────────────────────────────────
    st.subheader("Raw Metrics JSON")
    st.json(m)

    st.subheader("Metrics Table")
    df_table = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in m.items() if not isinstance(v, list)]
    )
    st.dataframe(df_table)

    # ─────────────────────────────────────────────
    # 6) Combined chart via Altair (optional)
    # ─────────────────────────────────────────────
    # Uncomment to use an Altair chart with combined bar, line, etc.
    import altair as alt

    combined = (
        pd.concat([df_price, df_pop], axis=1)
        .reset_index()
        .melt("timestamp", var_name="series", value_name="value")
    )
    fig_alt = (
        alt.Chart(combined)
        .mark_line()
        .encode(x="timestamp:T", y="value:Q", color="series:N")
        .interactive()
    )
    st.altair_chart(fig_alt, use_container_width=True)
