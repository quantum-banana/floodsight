import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from app.inference.contracts import (
    DetectionProvenance,
    DetectionResult,
    DetectorInferenceMode,
    ModelIdentity,
    NormalizedDetection,
    SegmentationProvenance,
    SegmentationResult,
    decode_mask,
)
from app.inference.coordinates import normalize_bbox
from app.inference.model_registry import ModelRegistry, ResolvedModel
from app.inference.taxonomy import load_taxonomy

COCO_TO_FLOODSIGHT = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
}
VEHICLE_APPLICATION_CLASSES = {
    "car",
    "van",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "tricycle",
}
CONTAINMENT_NMS_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionRuntime(Protocol):
    device: str

    def load(self) -> None: ...

    def predict(
        self,
        frame_bgr: NDArray[np.uint8],
        *,
        inference_resolution: int | None = None,
    ) -> list[RawDetection]: ...


class YoloAdapter:
    def __init__(
        self,
        model: ResolvedModel,
        *,
        device: str = "auto",
        precision: str = "auto",
        inference_resolution: int = 768,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        aerial_inference_resolution: int = 1280,
        aerial_tile_overlap: float = 0.2,
        aerial_fusion_iou_threshold: float = 0.5,
        aerial_high_recall_resolution: int = 1280,
        aerial_high_recall_tile_overlap: float = 0.25,
        aerial_high_recall_person_confidence: float = 0.5,
        segformer_reinspection_enabled: bool = True,
        segformer_reinspection_padding: float = 0.75,
        segformer_reinspection_min_pixels: int = 24,
        runtime: DetectionRuntime | None = None,
    ) -> None:
        self.model = model
        self.taxonomy = load_taxonomy(
            model.taxonomy, expected_version=model.record.taxonomy_version
        )
        self.runtime = runtime or UltralyticsRuntime(
            model,
            device=device,
            precision=precision,
            inference_resolution=inference_resolution,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        if aerial_inference_resolution < 256:
            raise ValueError("aerial inference resolution must be at least 256")
        if not 0 <= aerial_tile_overlap < 1:
            raise ValueError("aerial tile overlap must be in [0, 1)")
        if not 0 < aerial_fusion_iou_threshold <= 1:
            raise ValueError("aerial fusion IoU threshold must be in (0, 1]")
        if aerial_high_recall_resolution < 256:
            raise ValueError("aerial high-recall resolution must be at least 256")
        if not 0 <= aerial_high_recall_tile_overlap < 1:
            raise ValueError("aerial high-recall overlap must be in [0, 1)")
        if not 0 < aerial_high_recall_person_confidence <= 1:
            raise ValueError("aerial high-recall person confidence must be in (0, 1]")
        if not 0 <= segformer_reinspection_padding <= 2:
            raise ValueError("SegFormer reinspection padding must be in [0, 2]")
        if segformer_reinspection_min_pixels <= 0:
            raise ValueError("SegFormer reinspection minimum pixels must be positive")
        self.inference_resolution = inference_resolution
        self.aerial_inference_resolution = aerial_inference_resolution
        self.aerial_tile_overlap = aerial_tile_overlap
        self.aerial_fusion_iou_threshold = aerial_fusion_iou_threshold
        self.aerial_high_recall_resolution = aerial_high_recall_resolution
        self.aerial_high_recall_tile_overlap = aerial_high_recall_tile_overlap
        self.aerial_high_recall_person_confidence = aerial_high_recall_person_confidence
        self.segformer_reinspection_enabled = segformer_reinspection_enabled
        self.segformer_reinspection_padding = segformer_reinspection_padding
        self.segformer_reinspection_min_pixels = segformer_reinspection_min_pixels
        self.loaded = False

    def load(self) -> None:
        ModelRegistry.verify_checkpoint(self.model)
        self.runtime.load()
        self.loaded = True

    def infer(
        self,
        frame_bgr: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
        detector_mode: DetectorInferenceMode = DetectorInferenceMode.STANDARD,
        segmentation: SegmentationResult | None = None,
    ) -> DetectionResult:
        if not self.loaded:
            raise RuntimeError("YOLO adapter is not loaded")
        height, width = frame_bgr.shape[:2]
        started = time.perf_counter()
        mode = DetectorInferenceMode(detector_mode)
        raw_detections = self._predict(frame_bgr, mode, segmentation)
        normalized: list[NormalizedDetection] = []
        for index, item in enumerate(raw_detections, start=1):
            application_class = self._map_class(item.class_name)
            if application_class is None:
                continue
            if (
                mode is DetectorInferenceMode.AERIAL_HIGH_RECALL
                and application_class == "person"
                and item.confidence < self.aerial_high_recall_person_confidence
            ):
                continue
            normalized.append(
                NormalizedDetection(
                    detection_id=f"{frame_id}-{index}",
                    application_class=application_class,
                    source_class=item.class_name,
                    source_class_id=item.class_id,
                    confidence=item.confidence,
                    source_confidence=item.confidence,
                    bbox=normalize_bbox(*item.xyxy, width, height),
                    source_frame_id=frame_id,
                )
            )
        record = self.model.record
        return DetectionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=width,
            source_height=height,
            model=ModelIdentity(
                model_id=record.model_id,
                architecture=record.architecture,
                version=record.version,
                checkpoint_sha256=record.checkpoint_sha256 or _checkpoint_hash(self.model),
            ),
            taxonomy_version=self.taxonomy.version,
            detections=normalized,
            inference_latency_ms=round((time.perf_counter() - started) * 1_000, 3),
            device=self.runtime.device,
            provenance_mode=DetectionProvenance(record.provenance.value),
            source_frame_id=frame_id,
        )

    def _predict(
        self,
        frame_bgr: NDArray[np.uint8],
        detector_mode: DetectorInferenceMode,
        segmentation: SegmentationResult | None,
    ) -> list[RawDetection]:
        full_frame = self.runtime.predict(
            frame_bgr,
            inference_resolution=self.inference_resolution,
        )
        if detector_mode is DetectorInferenceMode.STANDARD:
            return full_frame

        height, width = frame_bgr.shape[:2]
        combined = list(full_frame)
        tile_passes = [(2, 2, self.aerial_tile_overlap, self.aerial_inference_resolution)]
        if detector_mode is DetectorInferenceMode.AERIAL_HIGH_RECALL:
            tile_passes.append(
                (
                    3,
                    3,
                    self.aerial_high_recall_tile_overlap,
                    self.aerial_high_recall_resolution,
                )
            )
        for rows, columns, overlap, resolution in tile_passes:
            for tile in overlapping_tile_windows(
                width,
                height,
                rows=rows,
                columns=columns,
                overlap=overlap,
            ):
                crop = np.ascontiguousarray(frame_bgr[tile.y1 : tile.y2, tile.x1 : tile.x2])
                for detection in self.runtime.predict(
                    crop,
                    inference_resolution=resolution,
                ):
                    translated = translate_detection(detection, tile, width, height)
                    if translated is not None:
                        combined.append(translated)
        fused = class_aware_nms(
            combined,
            class_mapper=self._map_class,
            iou_threshold=self.aerial_fusion_iou_threshold,
        )
        if (
            detector_mode is DetectorInferenceMode.AERIAL_HIGH_RECALL
            and self.segformer_reinspection_enabled
            and segmentation is not None
        ):
            fused = class_aware_nms(
                fused + self._segformer_guided_reinspection(frame_bgr, segmentation, fused),
                class_mapper=self._map_class,
                iou_threshold=self.aerial_fusion_iou_threshold,
            )
        return fused

    def _segformer_guided_reinspection(
        self,
        frame_bgr: NDArray[np.uint8],
        segmentation: SegmentationResult,
        detections: list[RawDetection],
    ) -> list[RawDetection]:
        """Run at most one semantic crop; semantic evidence never creates a detection."""
        height, width = frame_bgr.shape[:2]
        if (
            segmentation.provenance_mode is not SegmentationProvenance.REAL_MODEL
            or segmentation.source_width != width
            or segmentation.source_height != height
        ):
            return []
        vehicle_statistic = next(
            (item for item in segmentation.class_statistics if item.class_name == "vehicle"),
            None,
        )
        if (
            vehicle_statistic is None
            or vehicle_statistic.pixel_count < self.segformer_reinspection_min_pixels
        ):
            return []
        class_map = np.asarray(decode_mask(segmentation.mask), dtype=np.uint8)
        vehicle_mask = (class_map == vehicle_statistic.class_id).astype(np.uint8)
        component_count, _, statistics, _ = cv2.connectedComponentsWithStats(
            vehicle_mask,
            connectivity=8,
        )
        components = sorted(
            (
                tuple(int(value) for value in statistics[index])
                for index in range(1, component_count)
                if int(statistics[index, cv2.CC_STAT_AREA])
                >= self.segformer_reinspection_min_pixels
                and int(statistics[index, cv2.CC_STAT_WIDTH]) >= 3
                and int(statistics[index, cv2.CC_STAT_HEIGHT]) >= 3
            ),
            key=lambda item: (-item[cv2.CC_STAT_AREA], item[1], item[0]),
        )
        vehicle_detections = [
            item
            for item in detections
            if self._map_class(item.class_name) in VEHICLE_APPLICATION_CLASSES
        ]
        for x, y, component_width, component_height, _ in components:
            if _has_detection_near_component(
                vehicle_detections,
                x=x,
                y=y,
                width=component_width,
                height=component_height,
            ):
                continue
            padding_x = max(24, round(component_width * self.segformer_reinspection_padding))
            padding_y = max(24, round(component_height * self.segformer_reinspection_padding))
            window = TileWindow(
                x1=max(0, x - padding_x),
                y1=max(0, y - padding_y),
                x2=min(width, x + component_width + padding_x),
                y2=min(height, y + component_height + padding_y),
            )
            crop = np.ascontiguousarray(
                frame_bgr[window.y1 : window.y2, window.x1 : window.x2]
            )
            accepted: list[RawDetection] = []
            for detection in self.runtime.predict(
                crop,
                inference_resolution=self.aerial_high_recall_resolution,
            ):
                if self._map_class(detection.class_name) is None:
                    continue
                translated = translate_detection(detection, window, width, height)
                if translated is not None:
                    accepted.append(translated)
            return accepted
        return []

    def _map_class(self, source_name: str) -> str | None:
        record = self.model.record
        if record.label_space == "COCO":
            return COCO_TO_FLOODSIGHT.get(source_name.lower())
        if record.label_space == "FLOODSIGHT_DETECTION_V1":
            if source_name not in self.taxonomy.by_name:
                raise ValueError(f"Unknown final detector label {source_name!r}")
            return source_name
        raise ValueError(f"Unsupported detector label space {record.label_space!r}")


def overlapping_tile_windows(
    width: int,
    height: int,
    *,
    rows: int,
    columns: int,
    overlap: float,
) -> tuple[TileWindow, ...]:
    """Return deterministic row-major overlapping windows covering the full frame."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if rows <= 0 or columns <= 0:
        raise ValueError("tile rows and columns must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("tile overlap must be in [0, 1)")

    x_ranges = _overlapping_axis(width, columns, overlap)
    y_ranges = _overlapping_axis(height, rows, overlap)
    return tuple(
        TileWindow(x1=x1, y1=y1, x2=x2, y2=y2)
        for y1, y2 in y_ranges
        for x1, x2 in x_ranges
    )


def _overlapping_axis(
    length: int,
    count: int,
    overlap: float,
) -> tuple[tuple[int, int], ...]:
    if count == 1:
        return ((0, length),)
    tile_size = min(length, math.ceil(length / (count - overlap * (count - 1))))
    maximum_start = length - tile_size
    starts = [round(index * maximum_start / (count - 1)) for index in range(count)]
    return tuple((start, min(length, start + tile_size)) for start in starts)


def translate_detection(
    detection: RawDetection,
    tile: TileWindow,
    frame_width: int,
    frame_height: int,
) -> RawDetection | None:
    """Map a tile-local box into clamped original-frame pixel coordinates."""
    x1, y1, x2, y2 = detection.xyxy
    translated = (
        min(float(frame_width), max(0.0, x1 + tile.x1)),
        min(float(frame_height), max(0.0, y1 + tile.y1)),
        min(float(frame_width), max(0.0, x2 + tile.x1)),
        min(float(frame_height), max(0.0, y2 + tile.y1)),
    )
    if translated[2] <= translated[0] or translated[3] <= translated[1]:
        return None
    return RawDetection(
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        xyxy=translated,
    )


def class_aware_nms(
    detections: list[RawDetection],
    *,
    class_mapper: Callable[[str], str | None],
    iou_threshold: float,
) -> list[RawDetection]:
    """Suppress duplicate people/vehicles across full-frame and tiled passes."""
    supported = [item for item in detections if class_mapper(item.class_name) is not None]
    supported.sort(key=lambda item: (-item.confidence, item.class_id, item.xyxy))
    kept: list[RawDetection] = []
    for candidate in supported:
        family = _suppression_family(class_mapper(candidate.class_name))
        duplicate = any(
            _suppression_family(class_mapper(existing.class_name)) == family
            and _boxes_duplicate(candidate.xyxy, existing.xyxy, iou_threshold)
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _suppression_family(application_class: str | None) -> str | None:
    if application_class in VEHICLE_APPLICATION_CLASSES:
        return "vehicle"
    return application_class


def _boxes_duplicate(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    iou_threshold: float,
) -> bool:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    iou = intersection / union if union > 0 else 0.0
    smaller_area = min(left_area, right_area)
    containment = intersection / smaller_area if smaller_area > 0 else 0.0
    return iou >= iou_threshold or containment >= CONTAINMENT_NMS_THRESHOLD


def _has_detection_near_component(
    detections: list[RawDetection],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bool:
    margin_x = max(8.0, width * 0.25)
    margin_y = max(8.0, height * 0.25)
    x1, y1 = x - margin_x, y - margin_y
    x2, y2 = x + width + margin_x, y + height + margin_y
    return any(
        x1 <= (item.xyxy[0] + item.xyxy[2]) / 2 <= x2
        and y1 <= (item.xyxy[1] + item.xyxy[3]) / 2 <= y2
        for item in detections
    )


def _checkpoint_hash(model: ResolvedModel) -> str | None:
    path = model.checkpoint
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UltralyticsRuntime:
    def __init__(
        self,
        model: ResolvedModel,
        *,
        device: str,
        precision: str,
        inference_resolution: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.resolved = model
        self.requested_device = device
        self.precision = precision
        self.inference_resolution = inference_resolution
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = "unavailable"
        self._model: object | None = None
        self._half = False

    def load(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Torch and Ultralytics are required for real YOLO inference"
            ) from exc
        checkpoint = self.resolved.checkpoint
        if checkpoint is None:
            raise FileNotFoundError("YOLO checkpoint path is not configured")
        requested = self.requested_device.lower()
        if requested == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = requested
        self._half = self.device.startswith("cuda") and self.precision in {"auto", "float16"}
        torch.use_deterministic_algorithms(True, warn_only=True)
        self._model = YOLO(str(checkpoint), task="detect")

    def predict(
        self,
        frame_bgr: NDArray[np.uint8],
        *,
        inference_resolution: int | None = None,
    ) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("YOLO runtime is not loaded")
        precision_arguments = {"quantize": "fp16"} if self._half else {}
        results = self._model.predict(
            source=frame_bgr,
            imgsz=inference_resolution or self.inference_resolution,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            augment=False,
            verbose=False,
            **precision_arguments,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        detections: list[RawDetection] = []
        if result.boxes is None:
            return detections
        for row in result.boxes:
            class_id = int(row.cls.item())
            raw_name = names[class_id] if isinstance(names, dict) else names[class_id]
            xyxy = tuple(float(value) for value in row.xyxy[0].tolist())
            detections.append(
                RawDetection(
                    class_id=class_id,
                    class_name=str(raw_name),
                    confidence=float(row.conf.item()),
                    xyxy=xyxy,
                )
            )
        return detections
