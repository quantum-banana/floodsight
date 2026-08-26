from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from floodsight_data.errors import DatasetToolError

DATA_ROOT_ENV = "FLOODSIGHT_DATA_ROOT"
CACHE_ROOT_ENV = "FLOODSIGHT_DATA_CACHE"
LAYOUT_DIRECTORIES = (
    "raw",
    "interim/floodnet",
    "interim/rescuenet",
    "interim/visdrone_det",
    "processed/segmentation_v1",
    "processed/detection_v1",
    "manifests",
    "reports",
    "inspections",
    "locks",
)


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def resolve_data_root(value: str | Path | None = None, *, required: bool = True) -> Path | None:
    configured = value or os.environ.get(DATA_ROOT_ENV)
    if not configured:
        if required:
            raise DatasetToolError(
                f"Set {DATA_ROOT_ENV} or pass --data-root explicitly.",
                code="data_root_missing",
            )
        return None
    return _absolute(configured)


def resolve_cache_root(
    value: str | Path | None,
    *,
    data_root: Path | None,
    required: bool = False,
) -> Path | None:
    configured = value or os.environ.get(CACHE_ROOT_ENV)
    if configured:
        return _absolute(configured)
    if data_root is not None:
        return (data_root / ".cache").resolve()
    if required:
        raise DatasetToolError(
            f"Set {CACHE_ROOT_ENV} or pass --cache-root explicitly.",
            code="cache_root_missing",
        )
    return None


def ensure_contained(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise DatasetToolError(
            f"Refusing a path outside the intended root: {resolved_path}",
            code="unsafe_path",
        ) from exc
    if resolved_path == resolved_root:
        raise DatasetToolError(
            f"Refusing an operation against the root itself: {resolved_root}",
            code="unsafe_path",
        )
    return resolved_path


@dataclass(frozen=True, slots=True)
class DataPaths:
    root: Path
    cache: Path

    @classmethod
    def from_values(
        cls,
        data_root: str | Path | None,
        cache_root: str | Path | None = None,
    ) -> DataPaths:
        root = resolve_data_root(data_root)
        assert root is not None
        cache = resolve_cache_root(cache_root, data_root=root)
        assert cache is not None
        return cls(root=root, cache=cache)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def inspections(self) -> Path:
        return self.root / "inspections"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    def dataset_raw(self, dataset_id: str) -> Path:
        return self.raw / dataset_id

    def dataset_interim(self, dataset_id: str) -> Path:
        return self.interim / dataset_id

    def ensure_layout(self, *, dry_run: bool = False) -> list[Path]:
        planned = [self.root / relative for relative in LAYOUT_DIRECTORIES]
        planned.append(self.cache)
        if not dry_run:
            for directory in planned:
                directory.mkdir(parents=True, exist_ok=True)
        return planned
