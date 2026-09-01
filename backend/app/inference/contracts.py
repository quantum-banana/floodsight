from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.schemas.base import ContractModel
from app.schemas.live_result import BoundingBox, FeatureProvenance

UnitInterval = Annotated[float, Field(ge=0, le=1)]


class SegmentationProvenance(StrEnum):
    REAL_MODEL = "REAL_MODEL"
    SIMULATED = "SIMULATED"


class DetectionProvenance(StrEnum):
    REAL_MODEL = "REAL_MODEL"
    PRETRAINED_FALLBACK = "PRETRAINED_FALLBACK"
    SIMULATED = "SIMULATED"


EvidenceSource = FeatureProvenance


class ModelIdentity(ContractModel):
    model_id: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    version: str = Field(min_length=1)
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MaskRun(ContractModel):
    class_id: int = Field(ge=0, le=255)
    length: int = Field(gt=0)


class SemanticMask(ContractModel):
    encoding: Literal["ROW_MAJOR_RLE"] = "ROW_MAJOR_RLE"
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    runs: list[MaskRun] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pixel_count(self) -> Self:
        if sum(run.length for run in self.runs) != self.width * self.height:
            raise ValueError("semantic mask runs must cover exactly width * height pixels")
        return self


class SegmentationClassStatistic(ContractModel):
    class_id: int = Field(ge=0, le=255)
    class_name: str = Field(min_length=1)
    pixel_count: int = Field(ge=0)
    coverage_percent: float = Field(ge=0, le=100)
    mean_confidence: UnitInterval


class SegmentationResult(ContractModel):
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    model: ModelIdentity
    taxonomy_version: str = Field(min_length=1)
    mask: SemanticMask
    class_statistics: list[SegmentationClassStatistic]
    inference_latency_ms: float = Field(ge=0)
    device: str = Field(min_length=1)
    provenance_mode: SegmentationProvenance
    source_frame_id: int | None = Field(default=None, ge=0)
    reused_from_previous: bool = False


class NormalizedDetection(ContractModel):
    detection_id: str = Field(min_length=1)
    application_class: str = Field(min_length=1)
    source_class: str = Field(min_length=1)
    source_class_id: int = Field(ge=0)
    confidence: UnitInterval
    bbox: BoundingBox


class DetectionResult(ContractModel):
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    model: ModelIdentity
    taxonomy_version: str = Field(min_length=1)
    detections: list[NormalizedDetection]
    inference_latency_ms: float = Field(ge=0)
    device: str = Field(min_length=1)
    provenance_mode: DetectionProvenance
    source_frame_id: int | None = Field(default=None, ge=0)
    reused_from_previous: bool = False


class SemanticEvidence(ContractModel):
    class_id: int = Field(ge=0, le=255)
    class_name: str = Field(min_length=1)
    coverage_percent: float = Field(ge=0, le=100)
    confidence: UnitInterval
    source: EvidenceSource


class FusedScene(ContractModel):
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    taxonomy_version: str = Field(min_length=1)
    semantic_mask: SemanticMask | None
    semantic_evidence: list[SemanticEvidence]
    detections: list[NormalizedDetection]
    flood_class_ids: list[int]
    pool_class_id: int | None
    provenance: list[EvidenceSource]

    @model_validator(mode="after")
    def keep_pool_separate(self) -> Self:
        if self.pool_class_id is not None and self.pool_class_id in self.flood_class_ids:
            raise ValueError("pool must never be included in FloodSight flood evidence")
        return self


def encode_mask(class_map: object) -> SemanticMask:
    """Encode a two-dimensional integer array without exposing NumPy in the contract."""
    import numpy as np

    array = np.asarray(class_map)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("class_map must be a non-empty two-dimensional array")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("class_map IDs must fit in uint8")
    flattened = array.astype(np.uint8, copy=False).reshape(-1)
    changes = np.flatnonzero(flattened[1:] != flattened[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [flattened.size]))
    runs = [
        MaskRun(class_id=int(flattened[start]), length=int(end - start))
        for start, end in zip(starts, ends, strict=True)
    ]
    return SemanticMask(width=int(array.shape[1]), height=int(array.shape[0]), runs=runs)


def decode_mask(mask: SemanticMask) -> object:
    """Decode a semantic contract mask into a transient uint8 NumPy array."""
    import numpy as np

    values = np.fromiter((run.class_id for run in mask.runs), dtype=np.uint8, count=len(mask.runs))
    lengths = np.fromiter((run.length for run in mask.runs), dtype=np.int64, count=len(mask.runs))
    return np.repeat(values, lengths).reshape(mask.height, mask.width)
