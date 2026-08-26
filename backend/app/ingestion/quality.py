import cv2

from app.ingestion.frame_packet import FramePacket
from app.schemas.ingestion import FrameQuality


def assess_frame_quality(
    packet: FramePacket,
    *,
    dark_threshold: float,
    bright_threshold: float,
    blur_threshold: float,
) -> FrameQuality:
    gray = cv2.cvtColor(packet.bgr, cv2.COLOR_BGR2GRAY)
    luminance = float(gray.mean())
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings: list[str] = []

    if luminance < dark_threshold:
        brightness_status = "DARK"
        warnings.append("Frame is unusually dark; downstream analysis may be unreliable.")
    elif luminance > bright_threshold:
        brightness_status = "BRIGHT"
        warnings.append("Frame is unusually bright; downstream analysis may be unreliable.")
    else:
        brightness_status = "NORMAL"

    if laplacian_variance < blur_threshold:
        sharpness_status = "BLURRY"
        warnings.append("Frame appears blurry; downstream analysis may be unreliable.")
    else:
        sharpness_status = "NORMAL"

    return FrameQuality(
        mean_luminance=round(luminance, 3),
        laplacian_variance=round(laplacian_variance, 3),
        brightness_status=brightness_status,
        sharpness_status=sharpness_status,
        warnings=warnings,
    )
