import pytest
from httpx import AsyncClient

from app.schemas.health import HealthResponse

pytestmark = pytest.mark.anyio


async def test_health_returns_expected_contract(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "floodsight-api",
        "version": "0.1.0",
    }
    HealthResponse.model_validate(response.json())
