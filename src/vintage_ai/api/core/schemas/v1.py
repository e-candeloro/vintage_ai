# src/vintage_ai/core/schemas/v1.py
from typing import List, Optional
from datetime import date
from pydantic import Field, field_validator
from .base import BaseSchema

# --- request ---


class CarQuery(BaseSchema):
    car_name: str = Field(..., max_length=60, examples=["Porsche 911"])

    @field_validator("car_name")
    @classmethod
    def clean_car_name(cls, v: str) -> str:
        return v.strip().lower()


# --- response (flexible wrapper) ---


class CarMetric(BaseSchema):
    metric: str  # e.g. "current_popularity"
    value: Optional[float]  # None when the calc fails


class CarSnapshot(BaseSchema):
    car_name: str
    as_of: date
    metrics: List[CarMetric]  # **flexible** – add/drop metrics at will
