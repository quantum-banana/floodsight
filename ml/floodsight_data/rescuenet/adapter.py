from pathlib import Path

from floodsight_data.common.discovery import DiscoveryResult, discover_segmentation_pairs


def discover(root: Path) -> DiscoveryResult:
    return discover_segmentation_pairs(root)
