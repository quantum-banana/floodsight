import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.inference.contracts import (
    DetectionProvenance,
    DetectionResult,
    ModelIdentity,
    NormalizedDetection,
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


@dataclass(frozen=True, slots=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class DetectionRuntime(Protocol):
    device: str

    def load(self) -> None: ...

    def predict(self, frame_bgr: NDArray[np.uint8]) -> list[RawDetection]: ...


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
        self.loaded = False

    def load(self) -> None:
        ModelRegistry.verify_checkpoint(self.model)
        self.runtime.load()
        self.loaded = True

    def infer(
        self, frame_bgr: NDArray[np.uint8], *, frame_id: int, timestamp_ms: int
    ) -> DetectionResult:
        if not self.loaded:
            raise RuntimeError("YOLO adapter is not loaded")
        height, width = frame_bgr.shape[:2]
        started = time.perf_counter()
        raw_detections = self.runtime.predict(frame_bgr)
        normalized: list[NormalizedDetection] = []
        for index, item in enumerate(raw_detections, start=1):
            application_class = self._map_class(item.class_name)
            if application_class is None:
                continue
            normalized.append(
                NormalizedDetection(
                    detection_id=f"{frame_id}-{index}",
                    application_class=application_class,
                    source_class=item.class_name,
                    source_class_id=item.class_id,
                    confidence=item.confidence,
                    bbox=normalize_bbox(*item.xyxy, width, height),
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

    def _map_class(self, source_name: str) -> str | None:
        record = self.model.record
        if record.label_space == "COCO":
            return COCO_TO_FLOODSIGHT.get(source_name.lower())
        if record.label_space == "FLOODSIGHT_DETECTION_V1":
            if source_name not in self.taxonomy.by_name:
                raise ValueError(f"Unknown final detector label {source_name!r}")
            return source_name
        raise ValueError(f"Unsupported detector label space {record.label_space!r}")


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

    def predict(self, frame_bgr: NDArray[np.uint8]) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("YOLO runtime is not loaded")
        precision_arguments = {"quantize": "fp16"} if self._half else {}
        results = self._model.predict(
            source=frame_bgr,
            imgsz=self.inference_resolution,
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
