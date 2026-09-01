"""FloodSight Phase 5 object-detection training infrastructure.

The package deliberately has no import-time dependency on PyTorch or
Ultralytics.  Dataset contracts, configuration validation, and run planning can
therefore be audited in the lightweight repository environment.
"""

from floodsight_detection.contract import (
    DETECTION_CLASSES,
    DatasetContract,
    freeze_dataset_contract,
    validate_dataset_contract,
)
from floodsight_detection.errors import DetectionInfrastructureError

__all__ = [
    "DETECTION_CLASSES",
    "DatasetContract",
    "DetectionInfrastructureError",
    "freeze_dataset_contract",
    "validate_dataset_contract",
]

__version__ = "0.1.0"
