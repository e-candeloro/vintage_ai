# src/vintage_ai/settings.py
from pathlib import Path
from typing import List, Any

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Core URLs / paths – override them at runtime through environment   #
    # variables or a .env file                                           #
    # ------------------------------------------------------------------ #
    base_url: AnyUrl = Field("http://127.0.0.1:8000", alias="BASE_URL")
    api_base_path: str = Field("/api", alias="API_BASE_PATH")

    # Project layout
    python_path: Path = Field("src", alias="PYTHONPATH")
    dataset_path: Path = Field(
        "data/processed/asset_classic_car_prices_with_popularity.csv",
        alias="DATASET_PATH",
    )

    # CORS (comma-separated env var is handy for Cloud deployments)
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------------ #
    # Pydantic settings config                                           #
    # - env_file loads python-dotenv under the hood                      #
    # - env_prefix="" keeps variable names exactly as in .env / OS       #
    # - extra="allow" means *unknown* vars are stored in .model_extra    #
    #   so you can still access them: settings.model_extra["FOO"]        #
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Optional: post-processing hook (e.g. split comma-separated origins)
    def __init__(self, **data: Any):
        super().__init__(**data)
        if isinstance(self.allowed_origins, str):
            self.allowed_origins = [o.strip() for o in self.allowed_origins.split(",")]


# 1-liner singleton you can import everywhere
settings = Settings()
