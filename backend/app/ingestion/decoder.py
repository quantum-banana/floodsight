import cv2
import numpy as np

from app.ingestion.frame_packet import FramePacket
from app.schemas.ingestion import FrameMetadata


class FrameDecodeError(ValueError):
    """Raised when an encoded image cannot be decoded into a three-channel BGR frame."""


def decode_jpeg(session_id: str, metadata: FrameMetadata, payload: bytes) -> FramePacket:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise FrameDecodeError("The JPEG payload could not be decoded as a BGR image.")
    return FramePacket(
        session_id=session_id,
        metadata=metadata,
        encoded_bytes=payload,
        bgr=decoded,
    )
