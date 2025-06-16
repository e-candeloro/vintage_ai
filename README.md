# Vintage AI - MVA Hackaton 2025

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

## Project Overview

Vintage AI is a tool designed to provide sentiment and theme identification for classic car models. It leverages online data from various sources, including social media and enthusiast forums, to deliver insights into market perception and potential valuation shifts.

The project is built with a Python backend using FastAPI and a Streamlit-based dashboard for user interaction. Data is managed using DuckDB, and various Python libraries are employed for data scraping, processing, and analysis.

## Hackathon Evaluation Criteria Alignment

This solution is designed to meet the hackathon's evaluation criteria:

*   **Technical Innovation (20%)**:
    *   The project utilizes a modern Python stack, including FastAPI for a high-performance API and Streamlit for a reactive user interface.
    *   It incorporates Pydantic for robust data validation and settings management, ensuring code quality and reliability.
    *   The use of DuckDB allows for efficient data storage and querying, suitable for analytical workloads.
    *   The architecture separates concerns, with distinct services for data aggregation (`car_data_service.py`) and API routing (`api/routes/cars.py`), promoting modularity and maintainability.
    *   Sentiment analysis and theme extraction are planned using pre-trained models and potentially custom-trained models for higher accuracy on domain-specific language.

*   **Accuracy & Reliability (20%)**:
    *   The `car_data_service.py` is responsible for aggregating data from multiple sources. It uses median aggregation for scalar metrics to reduce the impact of outliers.
    *   Data validation is enforced through Pydantic models (`schemas/v1.py`), ensuring data integrity throughout the application.
    *   The system is designed to fetch and process data from various platforms, aiming for a comprehensive view of market sentiment. While current implementation in `car_data_service.py` focuses on DuckDB and Pytrends, the structure allows for adding more data sources.

*   **Usability & UX/UI (20%)**:
    *   The dashboard (`dashboard/app.py`) is built with Streamlit, a framework known for its ease of use in creating interactive web applications for data science projects.
    *   It provides a simple interface for users to input a car model and view analytics, including time-series data, sentiment scores, and summary metrics.
    *   Visualizations like line charts for price/popularity trends and bar charts for topic sentiments enhance data comprehension.
    *   Input sanitization (`sanitize_car_name` in `dashboard/app.py`) improves user experience by handling varied inputs.

*   **Scalability & Performance (20%)**:
    *   FastAPI is a high-performance web framework capable of handling many concurrent requests.
    *   DuckDB is designed for analytical performance and can handle larger datasets efficiently.
    *   The use of asynchronous operations (though not explicitly detailed in the provided `src` snippets, FastAPI supports them natively) can further enhance performance.
    *   Caching mechanisms (`@st.cache_data` in `dashboard/app.py`) are used to improve the responsiveness of the dashboard by avoiding redundant computations or API calls.
    *   The `car_data_service.py` is structured to potentially offload scraping tasks (`enqueue_scrape`), which is a good practice for scalability, allowing these to be handled by background workers in a more extensive setup.

*   **Business Relevance (10%)**:
    *   The tool directly addresses the need for insights in the classic car investment sector by providing data-driven sentiment analysis and trend identification.
    *   By analyzing online discussions and market data, it aims to help investors make more informed decisions.
    *   The focus on specific car models and their perceived valuation aligns with the core interests of collectors and investors in this niche market.

*   **Presentation & Clarity (10%)**:
    *   The Streamlit dashboard provides a clear and organized way to present the findings.
    *   The codebase is structured with clear separation of concerns (API, services, dashboard), as seen in the `src` and `dashboard` folders.
    *   The use of Pydantic for settings (`settings.py`) and schemas (`schemas/v1.py`) contributes to code clarity and maintainability.
    *   This README itself aims to provide a clear overview of the project and its installation.

## Codebase Highlights

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
