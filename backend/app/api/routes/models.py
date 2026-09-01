from fastapi import APIRouter, Request

from app.schemas.model_status import ModelStatusResponse
from app.services.model_status import get_model_status

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Report model configuration state",
)
async def model_status(request: Request) -> ModelStatusResponse:
    pipeline = getattr(request.app.state, "inference_pipeline", None)
    return pipeline.status() if pipeline is not None else get_model_status()
