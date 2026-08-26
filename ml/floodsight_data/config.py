from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from floodsight_data.errors import DatasetToolError

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
DATASET_CONFIG_ROOT = PROJECT_ROOT / "configs" / "datasets"
TAXONOMY_ROOT = PROJECT_ROOT / "shared" / "taxonomy"
SCHEMA_ROOT = PROJECT_ROOT / "shared" / "schemas"


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise DatasetToolError(
            f"Unable to read configuration: {path}",
            code="configuration_unreadable",
        ) from exc
    except yaml.YAMLError as exc:
        raise DatasetToolError(
            f"Invalid YAML in {path}: {exc}",
            code="configuration_invalid",
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetToolError(
            f"Configuration must contain a YAML object: {path}",
            code="configuration_invalid",
        )
    return payload


def stable_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DatasetToolError(
            f"Path is outside the configured data root: {path}",
            code="path_outside_data_root",
        ) from exc
