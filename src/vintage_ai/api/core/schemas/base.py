# src/vintage_ai/core/schemas/base.py
from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,  # allow alias="carName"
        extra="forbid",  # crash on unknown keys
        from_attributes=True,
    )
