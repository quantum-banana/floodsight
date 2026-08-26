from pathlib import Path

from numpy.typing import NDArray

from floodsight_data.common.images import read_source_mask, source_label_inventory
from floodsight_data.taxonomy import MappingTable


def parse_mask(path: Path, mapping: MappingTable) -> NDArray:
    return read_source_mask(path, mapping)


def inventory_mask(path: Path, mapping: MappingTable) -> dict[str, object]:
    return source_label_inventory(path, mapping)
