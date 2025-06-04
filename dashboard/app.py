import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from project root (only needed if not already loaded elsewhere)
load_dotenv()

# Load base path from environment
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_BASE_PATH = os.getenv("API_BASE_PATH", "/api")
BASE_ENDPOINT = f"{BASE_URL}{API_BASE_PATH}"


@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_api_data(endpoint: str):
    url = f"{BASE_ENDPOINT}{endpoint}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


st.title("Motor Valley Fest API 🎉")

if st.button("Fetch API Data"):
    st.subheader("GET /api/")
    root_data = fetch_api_data("/")
    st.json(root_data)

    st.subheader("GET /api/health")
    health_data = fetch_api_data("/health")
    st.json(health_data)
