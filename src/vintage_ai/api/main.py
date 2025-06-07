# src/vintage_ai/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vintage_ai.settings import settings
from vintage_ai.api.routes import cars


def create_app() -> FastAPI:
    app = FastAPI(title="VintageAI Hackathon API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(cars.router, prefix=settings.api_base_path)

    return app


app = create_app()
