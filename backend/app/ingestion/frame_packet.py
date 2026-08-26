from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.schemas.ingestion import FrameMetadata


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One transient decoded frame. Packets are never stored by the session manager."""

    session_id: str
    metadata: FrameMetadata
    encoded_bytes: bytes
    bgr: NDArray[np.uint8]
