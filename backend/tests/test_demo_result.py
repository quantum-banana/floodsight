import pytest
from httpx import AsyncClient

from app.schemas.live_result import DataOrigin, LiveResult

pytestmark = pytest.mark.anyio


async def test_demo_result_is_typed_and_explicitly_simulated(client: AsyncClient) -> None:
    response = await client.get("/api/demo/live-result")

    assert response.status_code == 200
    result = LiveResult.model_validate(response.json())
    assert result.data_origin is DataOrigin.DEMO_SIMULATED
    assert result.system_status.segmentation_model.value == "not_configured"
    assert result.system_status.detection_model.value == "not_configured"
    assert all(
        metric.data_origin is DataOrigin.DEMO_SIMULATED
        for metric in (
            result.statistics.flooded_area_percent,
            result.statistics.people_detected,
            result.statistics.vehicles_detected,
            result.statistics.blocked_roads,
            result.statistics.damaged_buildings,
        )
    )


async def test_unknown_route_uses_structured_error_response(client: AsyncClient) -> None:
    response = await client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "http_error",
            "message": "Not Found",
            "details": [],
        }
    }
