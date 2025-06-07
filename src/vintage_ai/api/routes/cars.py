# src/vintage_ai/api/routes/cars.py
from fastapi import APIRouter
from vintage_ai.api.core.schemas.v1 import CarQuery, CarSnapshot
from vintage_ai.services.car_data_service import load_snapshot


def enqueue_scrape(car_name: str) -> None:
    # TODO: integrate Celery / RQ / asyncio task that runs your scraper
    print(f"[SCRAPER] queued job for '{car_name}'")


router = APIRouter(prefix="/cars", tags=["Cars"])


@router.post("/snapshot", response_model=CarSnapshot)
def snapshot(payload: CarQuery) -> CarSnapshot:
    return load_snapshot(payload.car_name, on_missing=enqueue_scrape)
