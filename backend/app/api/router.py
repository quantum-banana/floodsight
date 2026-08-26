from fastapi import APIRouter

from app.api.routes import demo, health, models

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router, prefix="/api")
api_router.include_router(demo.router, prefix="/api")
api_router.include_router(demo.ws_router)
