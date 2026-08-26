from __future__ import annotations

import os
import shutil
from enum import StrEnum
from pathlib import Path

from floodsight_data.common.atomic import atomic_build
from floodsight_data.errors import DatasetToolError


class MaterializationStrategy(StrEnum):
    MANIFEST_ONLY = "manifest-only"
    HARDLINK = "hardlink"
    COPY = "copy"


def materialize_image(
    source: Path,
    destination: Path,
    *,
    strategy: MaterializationStrategy,
) -> Path:
    if strategy is MaterializationStrategy.MANIFEST_ONLY:
        return source
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    def build(temporary: Path) -> None:
        if strategy is MaterializationStrategy.HARDLINK:
            try:
                os.link(source, temporary)
                return
            except OSError:
                pass
        shutil.copy2(source, temporary)

    atomic_build(destination, build)
    return destination


def parse_materialization(value: str) -> MaterializationStrategy:
    try:
        return MaterializationStrategy(value)
    except ValueError as exc:
        raise DatasetToolError(
            f"Unknown materialization strategy: {value}",
            code="materialization_invalid",
        ) from exc
