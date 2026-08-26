from fastapi import APIRouter

from app.schemas.model_status import ModelStatusResponse
from app.services.model_status import get_model_status

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Report model configuration state",
)
async def model_status() -> ModelStatusResponse:
    return get_model_status()

