import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.schemas.live_result import LiveResult

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

