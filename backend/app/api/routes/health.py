from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="floodsight-api", version="0.1.0")
