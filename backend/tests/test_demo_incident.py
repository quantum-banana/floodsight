import pytest
from httpx import AsyncClient

from app.main import create_app
from app.schemas.incident import IncidentDetailResponse, IncidentListResponse, IncidentReport
from app.schemas.live_result import DataOrigin, LiveResult

pytestmark = pytest.mark.anyio


async def test_demo_incident_list_and_detail(client: AsyncClient) -> None:
    list_response = await client.get("/api/demo/incidents")
    detail_response = await client.get("/api/demo/incidents/FS-001")

    assert list_response.status_code == 200
    incident_list = IncidentListResponse.model_validate(list_response.json())
    assert [incident.incident_id for incident in incident_list.incidents] == ["FS-001"]
    assert incident_list.data_origin is DataOrigin.DEMO_SIMULATED

    assert detail_response.status_code == 200
    detail = IncidentDetailResponse.model_validate(detail_response.json())
    assert detail.incident.title == "Riverside Ward Flood Response"
    assert detail.snapshot_count == 6
    assert detail.initial_snapshot.snapshot_index == 0
    assert detail.latest_snapshot.snapshot_index == 5


async def test_demo_report_uses_latest_snapshot(client: AsyncClient) -> None:
    response = await client.get("/api/demo/incidents/FS-001/report")

    assert response.status_code == 200
    report = IncidentReport.model_validate(response.json())
    assert report.severity.value == "CRITICAL"
    assert report.statistics.flooded_area_percent.value == 42
    assert report.highest_priority_zone_id == "ZONE-2"
    assert report.data_origin is DataOrigin.DEMO_SIMULATED


async def test_unknown_demo_incident_uses_structured_error(client: AsyncClient) -> None:
    response = await client.get("/api/demo/incidents/FS-404")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "incident_not_found",
            "message": "Demo incident 'FS-404' was not found.",
            "details": [],
        }
    }


@pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient` is deprecated")
def test_websocket_first_message_and_deterministic_sequence() -> None:
    from fastapi.testclient import TestClient

    with (
        TestClient(create_app()) as websocket_client,
        websocket_client.websocket_connect(
            "/ws/demo/incidents/FS-001/live?interval_ms=0&loop=false"
        ) as websocket,
    ):
        messages = [LiveResult.model_validate(websocket.receive_json()) for _ in range(6)]

    assert messages[0].snapshot_index == 0
    assert messages[0].data_origin is DataOrigin.DEMO_SIMULATED
    assert [message.snapshot_index for message in messages] == list(range(6))
    assert messages[-1].stream_state.value == "COMPLETE"
