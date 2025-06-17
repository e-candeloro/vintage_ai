# 🚗 Vintage AI - MVA Hackathon 2025 🧠

![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-yellow?logo=duckdb&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/github/license/e-candeloro/vintage_ai)
![Pre-commit](https://img.shields.io/badge/Pre--commit-enabled-green)

> 💡 AI-powered tool for understanding classic car market trends and sentiment analysis across the web.

## 📽️ Demo Video
[![Watch the demo](https://img.shields.io/badge/▶️%20Watch%20Demo-video-ff0000)](media/vintage_ai_demo.mp4)




## 🧠 Project Overview

Vintage AI scrapes social media, forums and Google Trends to deliver **sentiment analysis** and **topic discovery** on classic-car models.

The project is built with a Python backend using FastAPI and a Streamlit-based dashboard for user interaction. Data is managed using DuckDB, and various Python libraries are employed for data scraping, processing, and analysis.

| Layer     | Tech                                                                                    |
| --------- | --------------------------------------------------------------------------------------- |
| Backend   | [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/latest/) |
| Frontend  | [Streamlit](https://streamlit.io/)                                                      |
| Storage   | [DuckDB](https://duckdb.org/)                                                           |
| Dev tools | [uv](https://github.com/astral-sh/uv), [pre-commit](https://pre-commit.com/)            |

## ✅ Hackathon Alignment

This solution is designed to meet the hackathon's evaluation criteria:

* 💡**Technical innovation** – modular Python stack, NLP pipelines  
* 📈 **Accuracy & reliability** – median aggregation, strict schema validation  
* 🎨 **Usability & UX** – one-click dashboard, interactive charts  
* ⚙️ **Scalability & performance** – DuckDB analytics, FastAPI async support, caching  
* 💼 **Business relevance** – investor-oriented insights for classic-car valuation  
* 📣 **Presentation & clarity** – clean architecture and documentation

## Installation

1. Install uv globally
2. Clone this repo

        git clone https://github.com/e-candeloro/vintage_ai.git

3. Go to the repo directory

        cd vintage_ai

4. Create and install dependencies
    
        uv sync

5. Install and test the pre-commit hooks

        pre-commit install
        pre-commit run --all-files

6. Rename the `.env.example` to `.env`. Inside this file there will be important environment settings for the program to use.  

7. Run the backend
   
        PYTHONPATH=src uvicorn vintage_ai.api.main:app --reload

8. Open a new shell an run the streamlit front-end:

        streamlit run dashboard/app.py

This will install all the dependencies, spin the FastAPI backend at `http://127.0.0.1:8000` and start the Streamlit dashboard at ` http://localhost:8501`




## 🧑‍💻 Codebase Highlights

*   **`src/vintage_ai/`**: Contains the core backend logic.
    *   **`api/`**: Defines the FastAPI application, including routes (`routes/cars.py`) and core schemas (`core/schemas/v1.py`).
        *   `main.py`: Sets up the FastAPI application and middleware.
        *   `routes/cars.py`: Handles API requests related to car data, currently providing a snapshot endpoint.
        *   `core/schemas/v1.py`: Defines Pydantic models for request and response data structures, ensuring type safety and validation.
    *   **`services/`**: Houses business logic.
        *   `car_data_service.py`: Contains functions to fetch, aggregate, and process car-related data from DuckDB and potentially other sources like Google Trends (via `fetch_trends_global`). It aggregates metrics from different platforms and stores/retrieves overall snapshots.
        *   `trends_services.py`: (Referenced in `car_data_service.py`) Likely contains logic for fetching trend data from external APIs like Google Trends.
    *   **`settings.py`**: Manages application settings using Pydantic's `BaseSettings`, allowing configuration via environment variables or a `.env` file. This is crucial for managing different environments (dev, prod) and sensitive information.

*   **`dashboard/app.py`**: A Streamlit application that serves as the user interface.
    *   It takes user input for a car model.
    *   Calls the backend API to fetch processed data.
    *   Displays various metrics and visualizations, including:
        *   Time-series charts for price and popularity.
        *   Correlation between price and popularity.
        *   Sentiment analysis (top topics and overall score).
        *   Summary metrics in a tabular and JSON format.
    *   Includes error handling for API calls and data processing.
    *   Uses caching to improve performance.
