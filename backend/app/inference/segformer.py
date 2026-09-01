import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from app.inference.contracts import (
    ModelIdentity,
    SegmentationClassStatistic,
    SegmentationProvenance,
    SegmentationResult,
    encode_mask,
)
from app.inference.model_registry import ModelRegistry, ResolvedModel
from app.inference.taxonomy import Taxonomy, load_taxonomy


@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    class_map: NDArray[np.uint8]
    confidence_map: NDArray[np.float32]


class SegmentationRuntime(Protocol):
    device: str

    def load(self) -> None: ...

    def predict(self, normalized_chw: NDArray[np.float32]) -> SegmentationPrediction: ...


class SegFormerAdapter:
    """Normalize a SegFormer runtime behind a stable, serializable application contract."""

    def __init__(
        self,
        model: ResolvedModel,
        *,
        device: str = "auto",
        precision: str = "auto",
        inference_resolution: int = 768,
        runtime: SegmentationRuntime | None = None,
    ) -> None:
        self.model = model
        self.taxonomy = load_taxonomy(
            model.taxonomy, expected_version=model.record.taxonomy_version
        )
        self.inference_resolution = inference_resolution
        self.runtime = runtime or TorchSegFormerRuntime(
            model,
            self.taxonomy,
            device=device,
            precision=precision,
            inference_resolution=inference_resolution,
        )
        self.loaded = False

    def load(self) -> None:
        ModelRegistry.verify_checkpoint(self.model)
        self.runtime.load()
        self.loaded = True

    def infer(
        self, frame_bgr: NDArray[np.uint8], *, frame_id: int, timestamp_ms: int
    ) -> SegmentationResult:
        if not self.loaded:
            raise RuntimeError("SegFormer adapter is not loaded")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("SegFormer input must be a three-channel BGR image")
        source_height, source_width = frame_bgr.shape[:2]
        started = time.perf_counter()
        resized = cv2.resize(
            frame_bgr,
            (self.inference_resolution, self.inference_resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        normalized = np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1))
        prediction = self.runtime.predict(normalized)
        class_map = cv2.resize(
            prediction.class_map,
            (source_width, source_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8, copy=False)
        confidence = cv2.resize(
            prediction.confidence_map,
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        statistics: list[SegmentationClassStatistic] = []
        for item in self.taxonomy.classes:
            selected = class_map == item.class_id
            pixel_count = int(np.count_nonzero(selected))
            statistics.append(
                SegmentationClassStatistic(
                    class_id=item.class_id,
                    class_name=item.name,
                    pixel_count=pixel_count,
                    coverage_percent=round(pixel_count * 100 / class_map.size, 6),
                    mean_confidence=(
                        round(float(confidence[selected].mean()), 6) if pixel_count else 0.0
                    ),
                )
            )
        record = self.model.record
        return SegmentationResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=source_width,
            source_height=source_height,
            model=ModelIdentity(
                model_id=record.model_id,
                architecture=record.architecture,
                version=record.version,
                checkpoint_sha256=record.checkpoint_sha256 or _checkpoint_hash(self.model),
            ),
            taxonomy_version=self.taxonomy.version,
            mask=encode_mask(class_map),
            class_statistics=statistics,
            inference_latency_ms=round((time.perf_counter() - started) * 1_000, 3),
            device=self.runtime.device,
            provenance_mode=SegmentationProvenance(record.provenance.value),
            source_frame_id=frame_id,
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


class TorchSegFormerRuntime:
    """Lazy Torch/Transformers implementation; optional dependencies stay out of fallback mode."""

    def __init__(
        self,
        model: ResolvedModel,
        taxonomy: Taxonomy,
        *,
        device: str,
        precision: str,
        inference_resolution: int,
    ) -> None:
        self.resolved = model
        self.taxonomy = taxonomy
        self.requested_device = device
        self.requested_precision = precision
        self.inference_resolution = inference_resolution
        self.device = "unavailable"
        self._torch: object | None = None
        self._model: object | None = None
        self._dtype: object | None = None

    def load(self) -> None:
        try:
            import torch
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "Torch and Transformers are required for real SegFormer inference"
            ) from exc
        device = _resolve_torch_device(torch, self.requested_device)
        dtype = _resolve_torch_dtype(torch, device, self.requested_precision)
        record = self.resolved.record
        checkpoint = self.resolved.checkpoint
        if checkpoint is None:
            raise FileNotFoundError("SegFormer checkpoint path is not configured")
        if record.checkpoint_format == "FLOODSIGHT_SEGFORMER_V3":
            configuration = _b2_configuration(SegformerConfig, self.taxonomy)
            model = SegformerForSemanticSegmentation(configuration)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("format") != (
                "floodsight-segformer-checkpoint-v3"
            ):
                raise ValueError("Unsupported FloodSight SegFormer checkpoint format")
            if payload.get("provenance") != "REAL_ML_OUTPUT":
                raise ValueError("SegFormer checkpoint is not labelled REAL_ML_OUTPUT")
            model.load_state_dict(payload["model"], strict=True)
        elif record.checkpoint_format == "HUGGINGFACE_DIRECTORY":
            model = SegformerForSemanticSegmentation.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=False
            )
            if model.config.num_labels != len(self.taxonomy.classes):
                raise ValueError("SegFormer label count does not match the registry taxonomy")
        else:
            raise ValueError(f"Unsupported SegFormer checkpoint format {record.checkpoint_format}")
        torch.use_deterministic_algorithms(True, warn_only=True)
        model.eval()
        model.requires_grad_(False)
        model.to(device=device, dtype=dtype)
        self._torch = torch
        self._model = model
        self._dtype = dtype
        self.device = str(device)

    def predict(self, normalized_chw: NDArray[np.float32]) -> SegmentationPrediction:
        if self._torch is None or self._model is None or self._dtype is None:
            raise RuntimeError("SegFormer runtime is not loaded")
        torch = self._torch
        tensor = torch.from_numpy(normalized_chw).unsqueeze(0).to(self.device)
        tensor = tensor.to(dtype=self._dtype)
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type,
                dtype=self._dtype,
                enabled=device_type == "cuda" and self._dtype != torch.float32,
            ):
                logits = self._model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(self.inference_resolution, self.inference_resolution),
                mode="bilinear",
                align_corners=False,
            ).float()
            probabilities = torch.softmax(logits, dim=1)
            confidence, classes = probabilities.max(dim=1)
        return SegmentationPrediction(
            class_map=classes[0].byte().cpu().numpy(),
            confidence_map=confidence[0].cpu().numpy().astype(np.float32, copy=False),
        )


def _resolve_torch_device(torch: object, requested: str) -> object:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _resolve_torch_dtype(torch: object, device: object, requested: str) -> object:
    if getattr(device, "type", "cpu") != "cuda":
        return torch.float32
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if requested == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _b2_configuration(config_type: type, taxonomy: Taxonomy) -> object:
    names = [item.name for item in taxonomy.classes]
    return config_type(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=768,
        semantic_loss_ignore_index=255,
        reshape_last_stage=True,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={name: index for index, name in enumerate(names)},
    )
