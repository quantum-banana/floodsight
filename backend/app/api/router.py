from fastapi import APIRouter

from app.api.routes import demo, health, ingestion, models

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router, prefix="/api")
api_router.include_router(demo.router, prefix="/api")
api_router.include_router(demo.ws_router)
api_router.include_router(ingestion.router, prefix="/api")
api_router.include_router(ingestion.ws_router)
