from fastapi import APIRouter

from app.schemas.live_result import LiveResult
from app.services.demo_result import get_demo_live_result

router = APIRouter(prefix="/demo", tags=["demo"])


@router.get(
    "/live-result",
    response_model=LiveResult,
    summary="Return the explicitly simulated Phase 0 incident example",
)
async def demo_live_result() -> LiveResult:
    return get_demo_live_result()

