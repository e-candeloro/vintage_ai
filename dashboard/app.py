import re
import pandas as pd
import httpx
from datetime import date
import asyncio
import os
import time  # Optional: for simulating delay

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# Load base path from environment
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_BASE_PATH = os.getenv("API_BASE_PATH", "/api")
BASE_ENDPOINT = f"{BASE_URL}{API_BASE_PATH}"

ENDPOINT = f"{BASE_ENDPOINT}/cars/snapshot"


# … existing imports …

# allow letters, digits, space, hyphen, apostrophe
SAFE_CHARS_PATTERN = re.compile(r"[^a-z0-9\s\-']")


def sanitize_car_name(raw: str) -> str:
    """
    Return a lowercase, space-normalised, punctuation-stripped car name.

    Security notes
    --------------
    * Removes angle brackets, quotes, semicolons… – common vectors for XSS or
      header injection (even though we're sending JSON).
    * Collapses multiple spaces to one.
    * Strips leading/trailing whitespace.
    """
    cleaned = SAFE_CHARS_PATTERN.sub("", raw.lower())  # keep only safe chars
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # normalise spaces
    return cleaned


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_snapshot(car_name: str) -> dict:
    res = httpx.post(ENDPOINT, json={"car_name": car_name}, timeout=15)
    res.raise_for_status()
    return res.json()


st.title("Classic-Car Popularity Dashboard 🚗✨")

car = st.text_input(
    "Car model",
    "Ferrari Testarossa",
    help="Enter the car model you want to analyze. E.g. 'Ferrari Testarossa'",
)

if st.button("Analyze"):
    safe_car = sanitize_car_name(car)
    if not safe_car:
        st.warning(
            "Please enter a valid car name (letters, numbers, hyphen, apostrophe)."
        )
        st.stop()

    with st.spinner("Calling backend …"):
        try:
            data = fetch_snapshot(car)
        except httpx.HTTPStatusError as e:
            st.error(f"{e.response.status_code}: {e.response.json().get('detail')}")
            st.stop()

    st.success(f"Metrics for **{data['car_name']}**")
    cols = st.columns(len(data["metrics"]))
    for c, m in zip(cols, data["metrics"]):
        c.metric(m["metric"].replace("_", " ").title(), m["value"])

    st.bar_chart(pd.DataFrame({m["metric"]: [m["value"]] for m in data["metrics"]}).T)

    # raw data json
    st.subheader("Raw data")
    st.json(data)
    # raw data table
    st.subheader("Raw data table")
    df = pd.DataFrame(data["metrics"])
