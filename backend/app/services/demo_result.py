import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.schemas.live_result import LiveResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = PROJECT_ROOT / "shared" / "examples" / "live-result.sample.json"


@lru_cache
def _load_demo_live_result() -> LiveResult:
    with SAMPLE_PATH.open(encoding="utf-8") as sample_file:
        payload: dict[str, Any] = json.load(sample_file)
    return LiveResult.model_validate(payload)


def get_demo_live_result() -> LiveResult:
    return _load_demo_live_result().model_copy(deep=True)

