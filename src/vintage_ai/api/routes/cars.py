# src/vintage_ai/api/routes/cars.py

from fastapi import APIRouter
from vintage_ai.api.core.schemas.v1 import CarQuery, CarSnapshot, Metrics
from vintage_ai.services.car_data_service import (
    aggregate_snapshot,
    has_metrics_for_car,
)


def enqueue_scrape(car_name: str) -> None:
    print(f"[SCRAPER] queued job for '{car_name}'")


router = APIRouter(prefix="/cars", tags=["Cars"])


@router.post("/snapshot", response_model=CarSnapshot)
def snapshot(payload: CarQuery) -> CarSnapshot:
    car_id = payload.car_name.strip().lower()

    if has_metrics_for_car(car_id):
        return aggregate_snapshot(car_id)

    enqueue_scrape(car_id)

    # Return empty CarSnapshot so frontend can still render something
    return CarSnapshot(
        car_id=car_id,
        metrics=Metrics(),  # all optional fields = None
        history=[],
    )
