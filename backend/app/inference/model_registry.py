import hashlib
import json
import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from app.schemas.base import ContractModel


class ModelType(StrEnum):
    SEGMENTATION = "SEGMENTATION"
    DETECTION = "DETECTION"


class ModelLifecycleStatus(StrEnum):
    INTEGRATION = "INTEGRATION"
    CANDIDATE = "CANDIDATE"
    BEST = "BEST"
    FINAL = "FINAL"
    FALLBACK = "FALLBACK"


class RegistryProvenance(StrEnum):
    REAL_MODEL = "REAL_MODEL"
    PRETRAINED_FALLBACK = "PRETRAINED_FALLBACK"
    SIMULATED = "SIMULATED"


class ModelRecord(ContractModel):
    model_id: str = Field(min_length=1)
    model_type: ModelType
    architecture: str = Field(min_length=1)
    version: str = Field(min_length=1)
    checkpoint_path: str | None = None
    checkpoint_env: str | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_format: str = Field(min_length=1)
    taxonomy_path: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    source_training_identity: str = Field(min_length=1)
    status: ModelLifecycleStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    provenance: RegistryProvenance
    enabled: bool = False
    label_space: str | None = None

    @model_validator(mode="after")
    def validate_path_source(self) -> "ModelRecord":
        if self.checkpoint_path and self.checkpoint_env:
            raise ValueError("model record must use checkpoint_path or checkpoint_env, not both")
        if self.provenance is not RegistryProvenance.SIMULATED and not (
            self.checkpoint_path or self.checkpoint_env
        ):
            raise ValueError("non-simulated model record requires a checkpoint source")
        return self


class ModelRegistryDocument(ContractModel):
    schema_version: str = Field(min_length=1)
    models: list[ModelRecord]


class ResolvedModel:
    def __init__(self, record: ModelRecord, checkpoint: Path | None, taxonomy: Path) -> None:
        self.record = record
        self.checkpoint = checkpoint
        self.taxonomy = taxonomy


class ModelRegistry:
    def __init__(
        self,
        *,
        path: Path,
        project_root: Path,
        environment: Mapping[str, str | None] | None = None,
    ) -> None:
        self.path = path.resolve()
        self.project_root = project_root.resolve()
        self.environment = environment if environment is not None else os.environ
        self.document = self._read()

    def _read(self) -> ModelRegistryDocument:
        try:
            raw: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load model registry: {exc}") from exc
        return ModelRegistryDocument.model_validate(raw)

    def enabled(self, model_type: ModelType) -> list[ResolvedModel]:
        return [
            self.resolve(record)
            for record in self.document.models
            if record.enabled and record.model_type is model_type
        ]

    def resolve(self, record: ModelRecord) -> ResolvedModel:
        raw_checkpoint = (
            self.environment.get(record.checkpoint_env)
            if record.checkpoint_env
            else record.checkpoint_path
        )
        checkpoint = self._resolve_path(raw_checkpoint) if raw_checkpoint else None
        taxonomy = self._resolve_path(record.taxonomy_path)
        if taxonomy is None or not taxonomy.is_file():
            raise ValueError(f"Taxonomy file is unavailable for {record.model_id}")
        return ResolvedModel(record, checkpoint, taxonomy)

    def _resolve_path(self, raw: str | None) -> Path | None:
        if raw is None or not raw.strip():
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    @staticmethod
    def verify_checkpoint(model: ResolvedModel) -> None:
        path = model.checkpoint
        if path is None or not path.exists():
            raise FileNotFoundError(f"Checkpoint is unavailable for {model.record.model_id}")
        if model.record.checkpoint_format.endswith("DIRECTORY"):
            if not path.is_dir():
                raise ValueError(f"Checkpoint for {model.record.model_id} must be a directory")
            return
        if not path.is_file():
            raise ValueError(f"Checkpoint for {model.record.model_id} must be a file")
        expected = model.record.checkpoint_sha256
        if expected is not None:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Checkpoint SHA-256 mismatch for {model.record.model_id}")
