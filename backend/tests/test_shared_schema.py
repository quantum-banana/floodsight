import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.schemas.live_result import LiveResult
from app.services.demo_incident import get_demo_snapshots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "shared" / "schemas" / "live-result.schema.json"
SAMPLE_PATH = PROJECT_ROOT / "shared" / "examples" / "live-result.sample.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def test_shared_sample_validates_against_json_schema_and_pydantic() -> None:
    schema = _read_json(SCHEMA_PATH)
    sample = _read_json(SAMPLE_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(sample)
    LiveResult.model_validate(sample)


def test_every_demo_snapshot_validates_against_shared_schema() -> None:
    schema = _read_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)

    for snapshot in get_demo_snapshots():
        payload = snapshot.model_dump(mode="json")
        validator.validate(payload)
        LiveResult.model_validate(payload)


def test_demo_identifiers_are_stable_and_provenance_is_explicit() -> None:
    snapshots = get_demo_snapshots()

    seen_zone_names: dict[str, str] = {}
    seen_event_payloads: dict[str, str] = {}
    for snapshot in snapshots:
        assert snapshot.data_origin.value == "DEMO_SIMULATED"
        for zone in snapshot.zones:
            previous = seen_zone_names.setdefault(zone.zone_id, zone.display_name)
            assert previous == zone.display_name
            assert zone.data_origin.value == "DEMO_SIMULATED"
        for event in snapshot.events:
            previous = seen_event_payloads.setdefault(event.event_id, event.message)
            assert previous == event.message
            assert event.data_origin.value == "DEMO_SIMULATED"
