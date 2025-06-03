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

## Repo idea/Architecture from ChatGPT (template)
```
vintage-ai/
├── .env                       # Secrets & runtime knobs           (▶ docker-compose, Airflow)
├── .pre-commit-config.yaml    # Black, Ruff, isort, nbstripout   (▶ dev quality gate)
├── docker-compose.yml         # One-shot local stack             (▶ all services)
├── Dockerfile                 # Base image for Python jobs       (▶ Airflow + workers)
├── pyproject.toml             # Poetry deps & src-layout         (▶ every Python module)
├── README.md                  # Quick-start for mentors
│
├── dags/                      # Airflow DAGs  ← plugs into airflow webserver
│   └── update_pipeline.py
│
├── src/                       # Importable package: `vintage_ai.*`
│   ├── vintage_ai/
│   │   ├── __init__.py
│   │   ├── config.py          # Central settings (pydantic.BaseSettings)
│   │   ├── ingest/            # Scrapers & loaders
│   │   │   ├── twitter.py
│   │   │   └── forums.py
│   │   ├── etl/               # DuckDB / Polars transformations
│   │   │   ├── raw_to_parquet.py
│   │   │   └── build_features.py
│   │   ├── models/            # Training & inference helpers
│   │   │   ├── train.py
│   │   │   └── predict.py
│   │   ├── api/               # FastAPI application
│   │   │   ├── main.py        # `uvicorn vintage_ai.api.main:app`
│   │   │   └── deps.py        # lazy loaders & DI
│   │   └── dashboard/         # Streamlit UI
│   │       └── app.py
│   └── tests/                 # pytest specs  (`poetry run pytest`)
│
├── data/                      # Local/ephemeral – .gitignored
│   ├── raw/
│   ├── parquet/
│   ├── features/
│   └── models/
└── notebooks/                 # Exploratory; nothing imported
```

---

## 1.  Back-bone ideas (why this layout works)

| Guideline                      | Concrete choice                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **One real Python package**    | `src/vintage_ai/…` → absolute imports (`from vintage_ai.etl.build_features import run`) avoid PYTHONPATH chaos.                                         |
| **Config lives outside code**  | `config.py` wraps **Pydantic BaseSettings** so every service (scraper, ETL, API) reads the same `.env` keys yet keeps sensible defaults for local runs. |
| **Everything containerised**   | A single `Dockerfile` (slim Python 3.11 + build-arg POETRY\_VERSION) is shared by Airflow workers and the FastAPI service.                              |
| **Airflow DAG thinness**       | Each DAG task is a one-liner wrapper that calls a function already in `vintage_ai.<layer>`. Business logic stays testable/outside Airflow.              |
| **Data & artefacts colocated** | Parquet feature store and pickled/torch models all under `data/`. The path is an env variable so S3 can replace it later.                               |

---

## 2.  What every config / infra file does

| File                          | Purpose                               | Key fields (hackathon defaults)                                                                                               |
| ----------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **`.env`**                    | runtime secrets & knobs               | `DATA_DIR=./data` · `MLFLOW_TRACKING_URI=./mlruns` · `MINIO_ENDPOINT=http://minio:9000`                                       |
| **`pyproject.toml`**          | dependency pinning + package metadata | `[tool.poetry.dependencies] neuralforecast="^1.8.0" …`<br>`[tool.poetry.scripts] vintage-ai-api = "vintage_ai.api.main:main"` |
| **`docker-compose.yml`**      | local orchestration                   | Services: `airflow-webserver`, `airflow-scheduler`, `api`, `streamlit`, `minio`, `duckdb-web`, `redis`                        |
| **`Dockerfile`**              | unified build image                   | 1. layer: python:3.11-slim → poetry install<br>2. copies `src/`, exposes `uvicorn` entrypoint                                 |
| **`dags/update_pipeline.py`** | ETL + train orchestration             | DAG args pull dates & bucket paths from `vintage_ai.config.settings`                                                          |
| **`vintage_ai/config.py`**    | **single source of truth**            | `python class Settings(BaseSettings): data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))`                                 |
| **`.pre-commit-config.yaml`** | on-save hygiene                       | Black, Ruff (flake8), isort, detect-secrets, nbstripout                                                                       |
| **`.gitignore`**              | keep repo slim                        | `/data/*`, `.env`, `.venv`, `__pycache__`, `*.parquet`, `*.pkl`                                                               |
| **`README.md`**               | five-line quick-start                 | Includes: `make dev` → `docker compose up --build`                                                                            |

---

## 3.  How the folders talk to each other (data & import flow)

```
┌──────────────┐        .jsonl          ┌────────────┐
│ ingest/*.py  │ ───────▶  data/raw  ───▶ etl/*.py   │
└──────────────┘                        └────┬───────┘
                                              │  Parquet
                                              ▼
                                        data/features/          ┌─────────────┐
                                              │  df slice       │ models/*.py │
                                              ▼                 └────┬────────┘
                                        train_models.py              │  pickle / torchscript
                                              │                      ▼
┌───────────────────┐    REST               api/main.py  ◀── streamlit/app.py
│ FastAPI container │◀────── JSON ─────────────────┘
└───────────────────┘
```

* **Ingest layer** writes raw JSONL to `data/raw/<source>/…`.
* **ETL layer** uses **DuckDB** to read that JSONL and spit Parquet into `data/parquet`.
* **Feature builder** runs more DuckDB/Polars, outputting a tidy table in `data/features/features.parquet`.
* **Model trainer** reads that Parquet, trains an NHITS/BERTopic model, pickles to `data/models/<car>.pkl` and registers it in MLflow.
* **FastAPI** loads every `*.pkl` on start-up (or reload) into an in-memory dict.
* **Streamlit** just calls the API and visualises. No business logic lives in the UI.

---

## 4.  Tiny code examples to anchor the flow

```python
# vintage_ai/ingest/twitter.py
from pathlib import Path, datetime
import subprocess, json

def run(out_dir: Path, query: str, hours: int = 1):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M")
    outfile = out_dir / f"{ts}.jsonl"
    cmd = ["snscrape", "--jsonl", "--since", f"{hours}h", "twitter-search", query]
    with outfile.open("w") as f:
        subprocess.run(cmd, stdout=f, check=True)

# vintage_ai/etl/raw_to_parquet.py
import duckdb, os, glob
def run(raw_dir: str, pq_dir: str):
    con = duckdb.connect()
    for fp in glob.glob(f"{raw_dir}/*.jsonl"):
        con.execute(f"COPY (SELECT * FROM read_json_auto('{fp}')) "
                    f"TO '{pq_dir}/{os.path.basename(fp)}.parquet' (FORMAT PARQUET);")

# vintage_ai/api/main.py
from fastapi import FastAPI, HTTPException
from vintage_ai.models.predict import load_models, predict
from vintage_ai.etl.feature_utils import latest_features

app = FastAPI()
models = load_models()

@app.get("/forecast/{car}")
def forecast(car: str):
    feats = latest_features(car)
    if car not in models:
        raise HTTPException(404, f"Model for {car} not found")
    yhat = predict(models[car], feats)
    return {"car": car, "forecast": yhat, "ts": feats["timestamp"].tolist()}
```

### TL;DR

*Declare a single installable package under `src/`, keep config in `.env`/`config.py`, and let Docker Compose spin the lot.
Every folder serves exactly one purpose and exchanges data via Parquet or plain function calls, so refactors stay painless when your dataset or traffic explodes.*
