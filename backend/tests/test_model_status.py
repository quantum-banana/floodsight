import pytest
from httpx import AsyncClient

from app.schemas.model_status import ModelStatusResponse

pytestmark = pytest.mark.anyio


async def test_models_are_honestly_unconfigured(client: AsyncClient) -> None:
    response = await client.get("/api/models/status")

    assert response.status_code == 200
    assert response.json() == {
        "segmentation": {"status": "not_configured", "model": None},
        "detection": {"status": "not_configured", "model": None},
    }
    ModelStatusResponse.model_validate(response.json())
