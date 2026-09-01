import pytest
from httpx import AsyncClient

from app.schemas.model_status import ModelStatusResponse

pytestmark = pytest.mark.anyio


async def test_models_are_honestly_unconfigured(client: AsyncClient) -> None:
    response = await client.get("/api/models/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_state"] == "MODEL_UNAVAILABLE"
    assert payload["segmentation"]["status"] == "unavailable"
    assert payload["segmentation"]["mode"] == "UNAVAILABLE"
    assert payload["detection"]["status"] == "unavailable"
    assert payload["detection"]["mode"] == "UNAVAILABLE"
    assert "checkpoint" not in str(payload).lower()
    ModelStatusResponse.model_validate(response.json())
