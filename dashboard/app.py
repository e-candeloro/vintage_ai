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


@st.cache_data(ttl=3600)
def fetch_api_data(endpoint: str):
    url = f"{BASE_ENDPOINT}{endpoint}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


st.title("Motor Valley Fest API 🎉")

if st.button("Fetch API Data"):
    with st.spinner("Running backend pipeline... Please wait ⏳"):
        # Simulate a long backend operation (remove this in real usage)
        time.sleep(3)

        # Fetch from API
        root_data = fetch_api_data("/")
        health_data = fetch_api_data("/health")

    st.success("Done!")

    st.subheader(f"GET {API_BASE_PATH}/")
    st.json(root_data)

    st.subheader(f"GET {API_BASE_PATH}/health")
    st.json(health_data)
