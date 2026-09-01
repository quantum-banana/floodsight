from __future__ import annotations

import sys
from pathlib import Path

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
if str(SEGMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_ROOT))
