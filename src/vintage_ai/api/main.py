from fastapi import APIRouter, FastAPI

from vintage_ai.settings import settings

app = FastAPI(title="Motor Valley Fest Hackathon API")

# this will serve at "rooth_path/api_base_path" in our case "http://127.0.0.1:8000/api"
api_router = APIRouter(prefix=settings.api_base_path)


@api_router.get("/health")
async def health():
    return {"status": "ok"}


@api_router.get("/")
async def root():
    return {"message": "Welcome to the Motor Valley Fest API!"}


app.include_router(api_router)
