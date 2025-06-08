# sentiment_batch.py
from pathlib import Path
from collections import Counter
import pandas as pd
from tqdm import tqdm
import os
import duckdb  # DuckDB for database operations

# ← the helper from your module
from vintage_ai.services.sentiment_service import _load_sentiment_pipeline
from vintage_ai.api.core.schemas.v1 import Metrics  # ← Pydantic schema

# same default as before
# your DuckDB writer
from vintage_ai.services.db_service import save_platform_metrics

# get model path from env or use default using load_dotenv

from dotenv import load_dotenv

load_dotenv()
MODEL_PATH: str = os.getenv(
    "SENTIMENT_MODEL_PATH", "data/models/sentiment_analysis/tabularisai"
)
db_path: str = os.getenv("DUCKDB_PATH", "data.duckdb")  # default DuckDB path


def process_saved_instagram_folder(
    folder: str | Path,
    *,
    model_path: str = MODEL_PATH,
    batch_size: int = 8,
) -> None:
    """
    Walk a folder full of *.parquet files that already contain scraped Instagram
    data, aggregate sentiment + engagement for each file and persist the result
    via `save_platform_metrics()`.

    Assumed Parquet schema
    ----------------------
    Each row should have **at least** these columns:
        - comments        (list[str])
        - likes           (int)
        - shares          (int)
        - plays           (int)
        - collections     (int)

    Parameters
    ----------
    folder : Path-like
        Directory that holds the parquet files (one file per car/model).
    model_path : str, optional
        Hugging Face model directory or HF Hub ID for sentiment pipeline.
    batch_size : int, optional
        Batch size passed to the transformers pipeline.
    """
    folder = Path(folder)
    pipe = _load_sentiment_pipeline(model_path)
    with duckdb.connect(db_path) as con:  # <-- one writer for the loop

        for pq in tqdm(folder.glob("*.parquet"), desc="Processing Parquet files..."):
            if not pq.is_file():
                continue
            car_id: str = pq.stem  # filename minus “.parquet”
            df = pd.read_parquet(pq)

            rows = []
            for _, rec in tqdm(
                df.iterrows(), total=len(df), desc=f"{car_id} sentiment"
            ):
                # --- safe comment extraction -----------------------------------------
                comments = rec.get("comments")
                if comments is None:
                    continue

                if not isinstance(comments, list):  # PyArrow / NumPy -> list
                    comments = list(comments)

                # keep only non-blank strings
                comments = [
                    c.strip() for c in comments if isinstance(c, str) and c.strip()
                ]
                if not comments:
                    continue
                # ---------------------------------------------------------------------

                preds = pipe(comments, batch_size=batch_size, truncation=True)
                if not preds:  # extra safety
                    continue

                scores = [p["score"] for p in preds]
                labels = [p["label"].lower() for p in preds]
                total = len(preds)
                total_eng = (
                    rec.get("likes", 0)
                    + rec.get("shares", 0)
                    + rec.get("collections", 0)
                    + total
                )

                rows.append(
                    dict(
                        num_comments=total,
                        avg_sentiment_score=sum(scores) / total,
                        most_common_sentiment=Counter(labels).most_common(1)[0][0],
                        likes=rec.get("likes", 0),
                        shares=rec.get("shares", 0),
                        plays=rec.get("plays", 0),
                        collections=rec.get("collections", 0),
                        engagement_score=round(
                            (total_eng / max(rec.get("plays", 0), 1)) * 100
                        ),
                    )
                )
            if not rows:  # nothing to aggregate for this parquet
                continue

            agg = pd.DataFrame(rows)
            metrics_dict = agg.median(numeric_only=True).to_dict()

            # force integer cast for known int fields
            for key in [
                "likes",
                "shares",
                "plays",
                "collections",
                "num_comments",
                "engagement_score",
            ]:
                if key in metrics_dict:
                    metrics_dict[key] = int(metrics_dict[key])
            metrics_dict["most_common_sentiment"] = (
                agg["most_common_sentiment"].mode().iat[0]
            )
            metrics_dict["overall_sentiment_score"] = agg["avg_sentiment_score"].mean()

            save_platform_metrics(
                car_id=car_id,
                platform="instagram",
                m=Metrics(**metrics_dict),
            )
            print(f"Processed {car_id} with {len(rows)} comments, saved metrics.")


if __name__ == "__main__":
    # Example usage
    process_saved_instagram_folder(
        folder="/home/ettore/projects/hackathons/MVA_hackathon_2025/vintage_ai/data/features/ig",
        model_path=MODEL_PATH,
        batch_size=8,
    )
    print("Sentiment processing completed.")
