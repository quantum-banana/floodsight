from pathlib import Path

from floodsight_data.common.discovery import DiscoveryResult, discover_visdrone_pairs


def discover(root: Path) -> DiscoveryResult:
    return discover_visdrone_pairs(root)
