from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from floodsight_data.paths import DataPaths


@pytest.fixture
def data_paths(tmp_path: Path) -> DataPaths:
    paths = DataPaths(root=tmp_path / "data", cache=tmp_path / "cache")
    paths.ensure_layout()
    return paths


@pytest.fixture
def write_rgb_image() -> Callable[[Path, tuple[int, int], int], Path]:
    def write(path: Path, size: tuple[int, int] = (8, 6), value: int = 100) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (value, value + 10, value + 20)).save(path)
        return path

    return write


@pytest.fixture
def write_indexed_mask() -> Callable[[Path, np.ndarray], Path]:
    def write(path: Path, values: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values.astype(np.uint8), mode="L").save(path)
        return path

    return write
