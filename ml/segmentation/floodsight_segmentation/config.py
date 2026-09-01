"""Strict, content-addressed configuration for SegFormer training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, DependencyError

CONFIG_SCHEMA_VERSION = "segmentation-training-v2"
PRODUCTION_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
PRODUCTION_MODEL_REVISION = "de01bae28967510f9ddd496c60a969357195400c"
UPSTREAM_PYTORCH_MODEL_SHA256 = "187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3"
PRODUCTION_SAFETENSORS_PATH = Path(
    "/data/floodsight-workspace/floodsight-cache/ml/models/segformer-b2-ade512/"
    "de01bae28967510f9ddd496c60a969357195400c/safetensors/model.safetensors"
)
PRODUCTION_SAFETENSORS_SHA256 = (
    "4a7ab8f05afe62dfdd75338b7fc2eb10ad1347bf5ade78ca2109951f5c717b86"
)
PRODUCTION_PROVENANCE_PATH = PRODUCTION_SAFETENSORS_PATH.with_name("provenance.json")
PRODUCTION_PROVENANCE_SHA256 = (
    "197a2a29f580406fc7d606445ffcba93bcf5d76dde3d4be6e2391c23a0a27add"
)
STAGE10_CLASS_WEIGHT_REPORT_PATH = Path(
    "/data/floodsight-workspace/floodsight-datasets/reports/pretraining_gate/"
    "segmentation_stages09_10_20260831T131322Z_v1/stage10/"
    "stage10_post_sanitation_class_balance_and_weights.json"
)
STAGE10_CLASS_WEIGHT_REPORT_SHA256 = (
    "1cfcdacdb1f08254170c3aeba458cb4aa85cd3d663bd4d220ef9fb8354733869"
)
TARGET_TAXONOMY_VERSION = "segmentation-taxonomy-v2"
MANIFEST_FINGERPRINT_ALGORITHM = "sha256-canonical-manifest-identity-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TAXONOMY_PATH = REPOSITORY_ROOT / "shared/taxonomy/segmentation-taxonomy-v2.yaml"
TAXONOMY_SHA256 = "2e10de69f9920aa113ae65c77b275ca506ed27ea7369bd8c9067bd899eca9ccf"
MAPPING_ASSETS = {
    "floodnet": (
        REPOSITORY_ROOT / "shared/taxonomy/floodnet-mapping-v2.yaml",
        "fdfbbba84c1cf8ea0176429b8d236693030abc16452f507c94922cc2f0769760",
    ),
    "rescuenet": (
        REPOSITORY_ROOT / "shared/taxonomy/rescuenet-mapping-v2.yaml",
        "cca60ac9977f7c7c70d290231561d127f2b8350a2a1cc82d9fb851e9f800dd28",
    ),
}
SEGMENTATION_RUN_ROOT = Path(
    "/data/floodsight-workspace/floodsight-datasets/runs/segmentation"
)
SEGMENTATION_REAL_SMOKE_ROOT = Path(
    "/data/floodsight-workspace/floodsight-datasets/runs/segmentation-real-smoke"
)
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
TARGET_CLASS_NAMES = (
    "background_other",
    "water",
    "road_non_flooded",
    "road_flooded",
    "road_clear",
    "road_blocked",
    "building_non_flooded",
    "building_flooded",
    "building_no_damage",
    "building_minor_damage",
    "building_major_damage",
    "building_destroyed",
    "vehicle",
    "tree",
    "grass",
    "pool",
)
TARGET_DATASET_SUPPORT = {
    "floodnet": frozenset({0, 1, 2, 3, 6, 7, 12, 13, 14, 15}),
    "rescuenet": frozenset({0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 15}),
}
AUDITED_SOURCE_TO_TARGET_IDS = {
    "floodnet": {0: 0, 1: 7, 2: 6, 3: 3, 4: 2, 5: 1, 6: 13, 7: 12, 8: 15, 9: 14},
    "rescuenet": {0: 0, 1: 1, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12, 7: 4, 8: 5, 9: 13, 10: 15},
}
AUDITED_MAPPING_VERSIONS = {
    "floodnet": "floodnet-mapping-v2",
    "rescuenet": "rescuenet-mapping-v2",
}
FINAL_CLASS_WEIGHTS = (
    0.25,
    0.393489627669,
    1.002582787879,
    1.194090320344,
    0.491677030293,
    1.028437432792,
    1.202818025361,
    1.512492458798,
    0.795489806735,
    0.79946331758,
    0.997437070651,
    1.079834182931,
    2.04400435097,
    0.25,
    0.31241852671,
    3.548595849419,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    pretrained_model_name_or_path: str
    revision: str
    local_files_only: bool
    trust_remote_code: bool
    weights_format: str
    upstream_pytorch_model_sha256: str
    safetensors_path: Path
    safetensors_sha256: str
    provenance_path: Path
    provenance_sha256: str
    num_labels: int
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenAssetConfig:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class TaxonomyAssetsConfig:
    taxonomy: FrozenAssetConfig
    mappings: Mapping[str, FrozenAssetConfig]

    @property
    def hashes(self) -> dict[str, str]:
        assets = (self.taxonomy, *self.mappings.values())
        return dict(sorted((str(asset.path), asset.sha256) for asset in assets))


@dataclass(frozen=True, slots=True)
class DataConfig:
    taxonomy_version: str
    manifest_fingerprint_algorithm: str
    ignore_index: int
    allowed_datasets: tuple[str, ...]
    supported_class_ids: Mapping[str, frozenset[int]]
    train_splits: tuple[str, ...]
    validation_splits: tuple[str, ...]
    require_full_integrity: bool
    verify_sample_hashes: bool


@dataclass(frozen=True, slots=True)
class LossConfig:
    name: str
    class_weight_policy: str
    normalization: str
    class_weight_source_path: Path
    class_weight_source_sha256: str
    class_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    name: str
    replacement: bool
    num_samples_policy: str
    dataset_mix: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OutputConfig:
    run_root: Path
    real_smoke_root: Path
    new_run_policy: str
    resume_policy: str
    last_checkpoint_filename: str
    best_checkpoint_filename: str
    history_filename: str
    report_filename: str


@dataclass(frozen=True, slots=True)
class TransformConfig:
    height: int
    width: int
    train_scale: tuple[float, float]
    train_ratio: tuple[float, float]
    horizontal_flip_probability: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    amsgrad: bool
    maximize: bool
    foreach: bool
    capturable: bool
    differentiable: bool
    fused: bool
    decoder_learning_rate_multiplier: float


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str
    warmup_ratio: float
    power: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    precision: str
    gradient_clip_norm: float
    validate_every_epochs: int
    checkpoint_every_epochs: int
    monitor_metric: str
    maximize_metric: bool


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    seed: int
    deterministic_algorithms: bool
    cudnn_benchmark: bool


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    path: Path
    sha256: str
    schema_version: str
    frozen: bool
    taxonomy_assets: TaxonomyAssetsConfig
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    sampler: SamplerConfig
    output: OutputConfig
    transforms: TransformConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    reproducibility: ReproducibilityConfig


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Expected an object at {location}.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ConfigurationError(
            f"Invalid keys at {location}: missing={sorted(missing)}, extra={sorted(extra)}."
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"Expected a non-empty string at {location}.")
    return value


def _hash(value: Any, location: str) -> str:
    digest = _string(value, location).lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ConfigurationError(f"Expected a lowercase SHA-256 at {location}.")
    return digest


def _absolute_path(value: Any, location: str) -> Path:
    raw = Path(_string(value, location)).expanduser()
    if not raw.is_absolute():
        raise ConfigurationError(f"Expected an absolute path at {location}.")
    if raw.is_symlink():
        raise ConfigurationError(f"Frozen path must not be a symbolic link at {location}.")
    return raw.resolve()


def _frozen_asset(
    value: Mapping[str, Any],
    location: str,
    *,
    expected_path: Path,
    expected_sha256: str,
) -> FrozenAssetConfig:
    _exact_keys(value, {"path", "sha256"}, location)
    path = _absolute_path(value["path"], f"{location}.path")
    digest = _hash(value["sha256"], f"{location}.sha256")
    if path != expected_path.resolve() or digest != expected_sha256:
        raise ConfigurationError(f"Frozen taxonomy identity drifted at {location}.")
    if not path.is_file():
        raise ConfigurationError(f"Frozen taxonomy asset is missing: {path}")
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read frozen taxonomy asset: {path}") from exc
    if actual != digest:
        raise ConfigurationError(
            f"Frozen taxonomy asset SHA-256 mismatch for {path}: expected {digest}, found {actual}."
        )
    return FrozenAssetConfig(path=path, sha256=digest)


def _bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"Expected a boolean at {location}.")
    return value


def _int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"Expected an integer >= {minimum} at {location}.")
    return value


def _float(value: Any, location: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < minimum
    ):
        raise ConfigurationError(f"Expected a number >= {minimum} at {location}.")
    return float(value)


def _sequence(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"Expected a non-empty array at {location}.")
    return value


def _pair(value: Any, location: str, *, positive: bool = True) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError(f"Expected a two-item array at {location}.")
    first = _float(value[0], f"{location}[0]", minimum=0.0)
    second = _float(value[1], f"{location}[1]", minimum=0.0)
    if (positive and first <= 0) or second < first:
        raise ConfigurationError(f"Invalid ordered range at {location}.")
    return first, second


def _triple(value: Any, location: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigurationError(f"Expected a three-item array at {location}.")
    return tuple(_float(item, f"{location}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration: {path}") from exc


def load_config(path: Path) -> SegmentationConfig:
    """Load and fully validate the frozen YAML configuration."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in dependency-free CLI use
        raise DependencyError("PyYAML is required to load a segmentation configuration.") from exc
    path = path.expanduser().resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to parse configuration: {path}") from exc
    root = _mapping(payload, "<root>")
    _exact_keys(
        root,
        {
            "schema_version",
            "frozen",
            "taxonomy_assets",
            "model",
            "data",
            "loss",
            "sampler",
            "output",
            "transforms",
            "optimizer",
            "scheduler",
            "training",
            "reproducibility",
        },
        "<root>",
    )
    schema_version = _string(root["schema_version"], "schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(f"Unsupported configuration schema {schema_version!r}.")
    frozen = _bool(root["frozen"], "frozen")
    if not frozen:
        raise ConfigurationError("Real training requires a configuration marked frozen: true.")

    taxonomy_raw = _mapping(root["taxonomy_assets"], "taxonomy_assets")
    _exact_keys(
        taxonomy_raw,
        {"taxonomy_path", "taxonomy_sha256", "mappings"},
        "taxonomy_assets",
    )
    taxonomy = _frozen_asset(
        {
            "path": taxonomy_raw["taxonomy_path"],
            "sha256": taxonomy_raw["taxonomy_sha256"],
        },
        "taxonomy_assets.taxonomy",
        expected_path=TAXONOMY_PATH,
        expected_sha256=TAXONOMY_SHA256,
    )
    mappings_raw = _mapping(taxonomy_raw["mappings"], "taxonomy_assets.mappings")
    if set(mappings_raw) != set(MAPPING_ASSETS):
        raise ConfigurationError(
            "taxonomy_assets.mappings must define exactly floodnet and rescuenet."
        )
    mappings = {
        dataset_id: _frozen_asset(
            _mapping(mappings_raw[dataset_id], f"taxonomy_assets.mappings.{dataset_id}"),
            f"taxonomy_assets.mappings.{dataset_id}",
            expected_path=MAPPING_ASSETS[dataset_id][0],
            expected_sha256=MAPPING_ASSETS[dataset_id][1],
        )
        for dataset_id in MAPPING_ASSETS
    }
    for dataset_id, asset in mappings.items():
        try:
            mapping_payload = yaml.safe_load(asset.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Unable to parse frozen mapping asset: {asset.path}") from exc
        mapping_root = _mapping(mapping_payload, f"mapping_asset.{dataset_id}")
        if (
            mapping_root.get("dataset_id") != dataset_id
            or mapping_root.get("task_type") != "SEMANTIC_SEGMENTATION"
            or mapping_root.get("mapping_version") != AUDITED_MAPPING_VERSIONS[dataset_id]
            or mapping_root.get("taxonomy_version") != TARGET_TAXONOMY_VERSION
        ):
            raise ConfigurationError(f"Frozen mapping header drifted for {dataset_id}.")
        raw_entries = mapping_root.get("mappings")
        if not isinstance(raw_entries, list):
            raise ConfigurationError(f"Frozen mapping rows are invalid for {dataset_id}.")
        observed_mapping: dict[int, int] = {}
        for index, raw_entry in enumerate(raw_entries):
            entry = _mapping(raw_entry, f"mapping_asset.{dataset_id}.mappings[{index}]")
            source_id = _int(
                entry.get("source_id"),
                f"mapping_asset.{dataset_id}.mappings[{index}].source_id",
            )
            target_id = _int(
                entry.get("target_id"),
                f"mapping_asset.{dataset_id}.mappings[{index}].target_id",
            )
            if entry.get("action") != "MAP" or source_id in observed_mapping:
                raise ConfigurationError(f"Frozen mapping row drifted for {dataset_id}.")
            observed_mapping[source_id] = target_id
        if observed_mapping != AUDITED_SOURCE_TO_TARGET_IDS[dataset_id]:
            raise ConfigurationError(f"Frozen source-to-target mapping drifted for {dataset_id}.")
    taxonomy_assets = TaxonomyAssetsConfig(taxonomy=taxonomy, mappings=mappings)

    model_raw = _mapping(root["model"], "model")
    _exact_keys(
        model_raw,
        {
            "pretrained_model_name_or_path",
            "revision",
            "local_files_only",
            "trust_remote_code",
            "weights_format",
            "upstream_pytorch_model_sha256",
            "safetensors_path",
            "safetensors_sha256",
            "provenance_path",
            "provenance_sha256",
            "num_labels",
            "class_names",
        },
        "model",
    )
    model_name = _string(
        model_raw["pretrained_model_name_or_path"], "model.pretrained_model_name_or_path"
    )
    if model_name != PRODUCTION_MODEL_ID:
        raise ConfigurationError(
            f"The frozen baseline must use {PRODUCTION_MODEL_ID!r}; found {model_name!r}."
        )
    revision = _string(model_raw["revision"], "model.revision")
    if revision != PRODUCTION_MODEL_REVISION:
        raise ConfigurationError(
            f"The frozen model revision must be {PRODUCTION_MODEL_REVISION}; found {revision!r}."
        )
    local_files_only = _bool(model_raw["local_files_only"], "model.local_files_only")
    trust_remote_code = _bool(model_raw["trust_remote_code"], "model.trust_remote_code")
    if not local_files_only or trust_remote_code:
        raise ConfigurationError(
            "The production model requires local_files_only: true and trust_remote_code: false."
        )
    weights_format = _string(model_raw["weights_format"], "model.weights_format")
    if weights_format != "safetensors":
        raise ConfigurationError("Only audited local safetensors weights are accepted.")
    upstream_sha256 = _string(
        model_raw["upstream_pytorch_model_sha256"],
        "model.upstream_pytorch_model_sha256",
    )
    if upstream_sha256 != UPSTREAM_PYTORCH_MODEL_SHA256:
        raise ConfigurationError("The pinned upstream PyTorch artifact SHA-256 does not match.")
    frozen_model_assets = {
        "safetensors": (
            _absolute_path(model_raw["safetensors_path"], "model.safetensors_path"),
            _hash(model_raw["safetensors_sha256"], "model.safetensors_sha256"),
            PRODUCTION_SAFETENSORS_PATH,
            PRODUCTION_SAFETENSORS_SHA256,
        ),
        "provenance": (
            _absolute_path(model_raw["provenance_path"], "model.provenance_path"),
            _hash(model_raw["provenance_sha256"], "model.provenance_sha256"),
            PRODUCTION_PROVENANCE_PATH,
            PRODUCTION_PROVENANCE_SHA256,
        ),
    }
    for label, (asset_path, asset_hash, expected_path, expected_hash) in (
        frozen_model_assets.items()
    ):
        if asset_path != expected_path.resolve() or asset_hash != expected_hash:
            raise ConfigurationError(f"Frozen model {label} identity drifted.")
        if asset_path.is_symlink() or not asset_path.is_file():
            raise ConfigurationError(f"Frozen model {label} must be a regular file: {asset_path}")
        actual_hash = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        if actual_hash != asset_hash:
            raise ConfigurationError(
                f"Frozen model {label} SHA-256 mismatch: expected {asset_hash}, "
                f"found {actual_hash}."
            )
    class_names = tuple(
        _string(item, f"model.class_names[{index}]")
        for index, item in enumerate(_sequence(model_raw["class_names"], "model.class_names"))
    )
    num_labels = _int(model_raw["num_labels"], "model.num_labels", minimum=2)
    if num_labels != len(class_names) or len(set(class_names)) != len(class_names):
        raise ConfigurationError("model.num_labels and unique model.class_names must agree.")
    if class_names != TARGET_CLASS_NAMES:
        raise ConfigurationError(
            "model.class_names must exactly match the frozen FloodSight taxonomy order."
        )
    model = ModelConfig(
        pretrained_model_name_or_path=model_name,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        weights_format=weights_format,
        upstream_pytorch_model_sha256=upstream_sha256,
        safetensors_path=frozen_model_assets["safetensors"][0],
        safetensors_sha256=frozen_model_assets["safetensors"][1],
        provenance_path=frozen_model_assets["provenance"][0],
        provenance_sha256=frozen_model_assets["provenance"][1],
        num_labels=num_labels,
        class_names=class_names,
    )

    data_raw = _mapping(root["data"], "data")
    _exact_keys(
        data_raw,
        {
            "taxonomy_version",
            "manifest_fingerprint_algorithm",
            "ignore_index",
            "allowed_datasets",
            "supported_class_ids",
            "train_splits",
            "validation_splits",
            "require_full_integrity",
            "verify_sample_hashes",
        },
        "data",
    )
    allowed = tuple(
        _string(item, f"data.allowed_datasets[{index}]")
        for index, item in enumerate(
            _sequence(data_raw["allowed_datasets"], "data.allowed_datasets")
        )
    )
    if len(set(allowed)) != len(allowed):
        raise ConfigurationError("data.allowed_datasets contains duplicates.")
    if allowed != tuple(TARGET_DATASET_SUPPORT):
        raise ConfigurationError("data.allowed_datasets must be exactly floodnet, rescuenet.")
    support_raw = _mapping(data_raw["supported_class_ids"], "data.supported_class_ids")
    if set(support_raw) != set(allowed):
        raise ConfigurationError(
            "data.supported_class_ids must define exactly the allowed datasets."
        )
    support: dict[str, frozenset[int]] = {}
    for dataset_id in allowed:
        ids = _sequence(support_raw[dataset_id], f"data.supported_class_ids.{dataset_id}")
        parsed = frozenset(
            _int(item, f"data.supported_class_ids.{dataset_id}[{index}]")
            for index, item in enumerate(ids)
        )
        if 0 not in parsed or any(class_id >= num_labels for class_id in parsed):
            raise ConfigurationError(
                f"Support for {dataset_id} must include background 0 and valid class IDs."
            )
        support[dataset_id] = parsed
    if support != TARGET_DATASET_SUPPORT:
        raise ConfigurationError(
            "data.supported_class_ids must exactly match the audited partial-supervision map."
        )
    ignore_index = _int(data_raw["ignore_index"], "data.ignore_index")
    if ignore_index != 255:
        raise ConfigurationError("data.ignore_index must remain the frozen value 255.")
    taxonomy_version = _string(data_raw["taxonomy_version"], "data.taxonomy_version")
    if taxonomy_version != TARGET_TAXONOMY_VERSION:
        raise ConfigurationError(
            f"data.taxonomy_version must be {TARGET_TAXONOMY_VERSION!r}."
        )
    fingerprint_algorithm = _string(
        data_raw["manifest_fingerprint_algorithm"],
        "data.manifest_fingerprint_algorithm",
    )
    if fingerprint_algorithm != MANIFEST_FINGERPRINT_ALGORITHM:
        raise ConfigurationError(
            "data.manifest_fingerprint_algorithm must use the canonical frozen formula."
        )
    data = DataConfig(
        taxonomy_version=taxonomy_version,
        manifest_fingerprint_algorithm=fingerprint_algorithm,
        ignore_index=ignore_index,
        allowed_datasets=allowed,
        supported_class_ids=support,
        train_splits=tuple(
            _string(item, f"data.train_splits[{index}]")
            for index, item in enumerate(_sequence(data_raw["train_splits"], "data.train_splits"))
        ),
        validation_splits=tuple(
            _string(item, f"data.validation_splits[{index}]")
            for index, item in enumerate(
                _sequence(data_raw["validation_splits"], "data.validation_splits")
            )
        ),
        require_full_integrity=_bool(
            data_raw["require_full_integrity"], "data.require_full_integrity"
        ),
        verify_sample_hashes=_bool(
            data_raw["verify_sample_hashes"], "data.verify_sample_hashes"
        ),
    )
    if not data.require_full_integrity or not data.verify_sample_hashes:
        raise ConfigurationError(
            "The frozen data path requires full manifest integrity and per-sample hashes."
        )
    if data.train_splits != ("train",) or data.validation_splits != ("val",):
        raise ConfigurationError(
            "The frozen data split roles must be train_splits=['train'] and "
            "validation_splits=['val']."
        )

    loss_raw = _mapping(root["loss"], "loss")
    _exact_keys(
        loss_raw,
        {
            "name",
            "class_weight_policy",
            "normalization",
            "class_weight_source_path",
            "class_weight_source_sha256",
            "class_weights",
        },
        "loss",
    )
    class_weight_source_path = _absolute_path(
        loss_raw["class_weight_source_path"], "loss.class_weight_source_path"
    )
    class_weight_source_sha256 = _hash(
        loss_raw["class_weight_source_sha256"], "loss.class_weight_source_sha256"
    )
    if (
        class_weight_source_path != STAGE10_CLASS_WEIGHT_REPORT_PATH.resolve()
        or class_weight_source_sha256 != STAGE10_CLASS_WEIGHT_REPORT_SHA256
    ):
        raise ConfigurationError("The frozen Stage-10 class-weight source identity drifted.")
    if class_weight_source_path.is_symlink() or not class_weight_source_path.is_file():
        raise ConfigurationError(
            "The frozen Stage-10 class-weight source must be a regular file: "
            f"{class_weight_source_path}"
        )
    try:
        class_weight_source_bytes = class_weight_source_path.read_bytes()
        class_weight_source_payload = json.loads(class_weight_source_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Unable to read the frozen Stage-10 class-weight source: {class_weight_source_path}"
        ) from exc
    actual_class_weight_source_sha256 = hashlib.sha256(class_weight_source_bytes).hexdigest()
    if actual_class_weight_source_sha256 != class_weight_source_sha256:
        raise ConfigurationError(
            "Frozen Stage-10 class-weight source SHA-256 mismatch: expected "
            f"{class_weight_source_sha256}, found {actual_class_weight_source_sha256}."
        )
    class_weight_source_root = _mapping(
        class_weight_source_payload, "loss.class_weight_source"
    )
    class_weight_proposal = _mapping(
        class_weight_source_root.get("class_weight_proposal"),
        "loss.class_weight_source.class_weight_proposal",
    )
    source_weights_raw = _sequence(
        class_weight_proposal.get("weights_target_id_order"),
        "loss.class_weight_source.class_weight_proposal.weights_target_id_order",
    )
    source_weights = tuple(
        _float(
            item,
            "loss.class_weight_source.class_weight_proposal."
            f"weights_target_id_order[{index}]",
            minimum=1e-12,
        )
        for index, item in enumerate(source_weights_raw)
    )
    if source_weights != FINAL_CLASS_WEIGHTS:
        raise ConfigurationError(
            "The Stage-10 class-weight source no longer contains the frozen target-ID vector."
        )
    weights_raw = _sequence(loss_raw["class_weights"], "loss.class_weights")
    class_weights = tuple(
        _float(item, f"loss.class_weights[{index}]", minimum=1e-12)
        for index, item in enumerate(weights_raw)
    )
    if len(class_weights) != num_labels:
        raise ConfigurationError(
            f"loss.class_weights must contain exactly {num_labels} positive values."
        )
    if class_weights != FINAL_CLASS_WEIGHTS:
        raise ConfigurationError(
            "loss.class_weights must exactly match the Stage-10 retained-training proposal."
        )
    loss = LossConfig(
        name=_string(loss_raw["name"], "loss.name"),
        class_weight_policy=_string(
            loss_raw["class_weight_policy"], "loss.class_weight_policy"
        ),
        normalization=_string(loss_raw["normalization"], "loss.normalization"),
        class_weight_source_path=class_weight_source_path,
        class_weight_source_sha256=class_weight_source_sha256,
        class_weights=class_weights,
    )
    if (
        loss.name != "dataset_masked_weighted_cross_entropy"
        or loss.class_weight_policy != "fixed_explicit"
        or loss.normalization != "sum_valid_pixel_weights"
    ):
        raise ConfigurationError("The frozen weighted partial-supervision loss policy drifted.")

    sampler_raw = _mapping(root["sampler"], "sampler")
    _exact_keys(
        sampler_raw,
        {"name", "replacement", "num_samples_policy", "dataset_mix"},
        "sampler",
    )
    mix_raw = _mapping(sampler_raw["dataset_mix"], "sampler.dataset_mix")
    if set(mix_raw) != set(allowed):
        raise ConfigurationError("sampler.dataset_mix must define exactly the allowed datasets.")
    mix = {
        key: _float(value, f"sampler.dataset_mix.{key}") for key, value in mix_raw.items()
    }
    if any(value <= 0 for value in mix.values()) or abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ConfigurationError("sampler.dataset_mix weights must be positive and sum to 1.0.")
    sampler = SamplerConfig(
        name=_string(sampler_raw["name"], "sampler.name"),
        replacement=_bool(sampler_raw["replacement"], "sampler.replacement"),
        num_samples_policy=_string(
            sampler_raw["num_samples_policy"], "sampler.num_samples_policy"
        ),
        dataset_mix=mix,
    )
    if (
        sampler.name != "dataset_balanced_weighted_random"
        or not sampler.replacement
        or sampler.num_samples_policy != "training_manifest_size"
        or mix != {"floodnet": 0.5, "rescuenet": 0.5}
    ):
        raise ConfigurationError("The frozen 50/50 dataset-balanced sampler policy drifted.")

    output_raw = _mapping(root["output"], "output")
    _exact_keys(
        output_raw,
        {
            "run_root",
            "real_smoke_root",
            "new_run_policy",
            "resume_policy",
            "last_checkpoint_filename",
            "best_checkpoint_filename",
            "history_filename",
            "report_filename",
        },
        "output",
    )
    output = OutputConfig(
        run_root=_absolute_path(output_raw["run_root"], "output.run_root"),
        real_smoke_root=_absolute_path(
            output_raw["real_smoke_root"], "output.real_smoke_root"
        ),
        new_run_policy=_string(output_raw["new_run_policy"], "output.new_run_policy"),
        resume_policy=_string(output_raw["resume_policy"], "output.resume_policy"),
        last_checkpoint_filename=_string(
            output_raw["last_checkpoint_filename"], "output.last_checkpoint_filename"
        ),
        best_checkpoint_filename=_string(
            output_raw["best_checkpoint_filename"], "output.best_checkpoint_filename"
        ),
        history_filename=_string(output_raw["history_filename"], "output.history_filename"),
        report_filename=_string(output_raw["report_filename"], "output.report_filename"),
    )
    if (
        output.run_root != SEGMENTATION_RUN_ROOT
        or output.real_smoke_root != SEGMENTATION_REAL_SMOKE_ROOT
        or output.new_run_policy != "exclusive_direct_child"
        or output.resume_policy != "approved_run_last_only"
        or output.last_checkpoint_filename != "last.pt"
        or output.best_checkpoint_filename != "best.pt"
        or output.history_filename != "history.json"
        or output.report_filename != "training-report.json"
    ):
        raise ConfigurationError("The frozen output/run/resume policy drifted.")

    transforms_raw = _mapping(root["transforms"], "transforms")
    _exact_keys(
        transforms_raw,
        {
            "height",
            "width",
            "train_scale",
            "train_ratio",
            "horizontal_flip_probability",
            "image_mean",
            "image_std",
        },
        "transforms",
    )
    flip = _float(
        transforms_raw["horizontal_flip_probability"],
        "transforms.horizontal_flip_probability",
    )
    if flip > 1:
        raise ConfigurationError("transforms.horizontal_flip_probability must be <= 1.")
    std = _triple(transforms_raw["image_std"], "transforms.image_std")
    if any(value <= 0 for value in std):
        raise ConfigurationError("transforms.image_std entries must be positive.")
    train_scale = _pair(transforms_raw["train_scale"], "transforms.train_scale")
    if train_scale[1] > 1:
        raise ConfigurationError(
            "transforms.train_scale is a source-area fraction and must be <= 1."
        )
    transforms = TransformConfig(
        height=_int(transforms_raw["height"], "transforms.height", minimum=1),
        width=_int(transforms_raw["width"], "transforms.width", minimum=1),
        train_scale=train_scale,
        train_ratio=_pair(transforms_raw["train_ratio"], "transforms.train_ratio"),
        horizontal_flip_probability=flip,
        image_mean=_triple(transforms_raw["image_mean"], "transforms.image_mean"),
        image_std=std,
    )

    optimizer_raw = _mapping(root["optimizer"], "optimizer")
    _exact_keys(
        optimizer_raw,
        {
            "name",
            "learning_rate",
            "weight_decay",
            "betas",
            "epsilon",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoder_learning_rate_multiplier",
        },
        "optimizer",
    )
    betas = _pair(optimizer_raw["betas"], "optimizer.betas", positive=False)
    if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ConfigurationError("optimizer.betas must lie in [0, 1).")
    optimizer = OptimizerConfig(
        name=_string(optimizer_raw["name"], "optimizer.name"),
        learning_rate=_float(
            optimizer_raw["learning_rate"], "optimizer.learning_rate", minimum=1e-12
        ),
        weight_decay=_float(optimizer_raw["weight_decay"], "optimizer.weight_decay"),
        betas=betas,
        epsilon=_float(optimizer_raw["epsilon"], "optimizer.epsilon", minimum=1e-12),
        amsgrad=_bool(optimizer_raw["amsgrad"], "optimizer.amsgrad"),
        maximize=_bool(optimizer_raw["maximize"], "optimizer.maximize"),
        foreach=_bool(optimizer_raw["foreach"], "optimizer.foreach"),
        capturable=_bool(optimizer_raw["capturable"], "optimizer.capturable"),
        differentiable=_bool(
            optimizer_raw["differentiable"], "optimizer.differentiable"
        ),
        fused=_bool(optimizer_raw["fused"], "optimizer.fused"),
        decoder_learning_rate_multiplier=_float(
            optimizer_raw["decoder_learning_rate_multiplier"],
            "optimizer.decoder_learning_rate_multiplier",
            minimum=1e-12,
        ),
    )
    if (
        optimizer.name != "adamw"
        or optimizer.epsilon != 1e-8
        or optimizer.amsgrad
        or optimizer.maximize
        or optimizer.foreach
        or optimizer.capturable
        or optimizer.differentiable
        or optimizer.fused
    ):
        raise ConfigurationError("The explicit audited AdamW policy drifted.")

    scheduler_raw = _mapping(root["scheduler"], "scheduler")
    _exact_keys(scheduler_raw, {"name", "warmup_ratio", "power"}, "scheduler")
    warmup = _float(scheduler_raw["warmup_ratio"], "scheduler.warmup_ratio")
    if warmup >= 1:
        raise ConfigurationError("scheduler.warmup_ratio must be < 1.")
    scheduler = SchedulerConfig(
        name=_string(scheduler_raw["name"], "scheduler.name"),
        warmup_ratio=warmup,
        power=_float(scheduler_raw["power"], "scheduler.power", minimum=1e-12),
    )
    if scheduler.name != "warmup_polynomial":
        raise ConfigurationError("Only the audited warmup_polynomial scheduler is supported.")

    training_raw = _mapping(root["training"], "training")
    _exact_keys(
        training_raw,
        {
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "precision",
            "gradient_clip_norm",
            "validate_every_epochs",
            "checkpoint_every_epochs",
            "monitor_metric",
            "maximize_metric",
        },
        "training",
    )
    precision = _string(training_raw["precision"], "training.precision")
    if precision != "bf16":
        raise ConfigurationError("The frozen single-H100 training precision must be bf16.")
    training = TrainingConfig(
        epochs=_int(training_raw["epochs"], "training.epochs", minimum=1),
        batch_size=_int(training_raw["batch_size"], "training.batch_size", minimum=1),
        gradient_accumulation_steps=_int(
            training_raw["gradient_accumulation_steps"],
            "training.gradient_accumulation_steps",
            minimum=1,
        ),
        num_workers=_int(training_raw["num_workers"], "training.num_workers"),
        precision=precision,
        gradient_clip_norm=_float(
            training_raw["gradient_clip_norm"], "training.gradient_clip_norm"
        ),
        validate_every_epochs=_int(
            training_raw["validate_every_epochs"], "training.validate_every_epochs", minimum=1
        ),
        checkpoint_every_epochs=_int(
            training_raw["checkpoint_every_epochs"],
            "training.checkpoint_every_epochs",
            minimum=1,
        ),
        monitor_metric=_string(training_raw["monitor_metric"], "training.monitor_metric"),
        maximize_metric=_bool(training_raw["maximize_metric"], "training.maximize_metric"),
    )
    if training.num_workers != 0:
        raise ConfigurationError(
            "The frozen 64-MiB-shm host policy requires training.num_workers: 0."
        )

    reproducibility_raw = _mapping(root["reproducibility"], "reproducibility")
    _exact_keys(
        reproducibility_raw,
        {"seed", "deterministic_algorithms", "cudnn_benchmark"},
        "reproducibility",
    )
    reproducibility = ReproducibilityConfig(
        seed=_int(reproducibility_raw["seed"], "reproducibility.seed"),
        deterministic_algorithms=_bool(
            reproducibility_raw["deterministic_algorithms"],
            "reproducibility.deterministic_algorithms",
        ),
        cudnn_benchmark=_bool(
            reproducibility_raw["cudnn_benchmark"], "reproducibility.cudnn_benchmark"
        ),
    )
    if reproducibility.deterministic_algorithms and reproducibility.cudnn_benchmark:
        raise ConfigurationError(
            "cudnn_benchmark must be false when deterministic_algorithms is true."
        )
    return SegmentationConfig(
        path=path,
        sha256=_sha256(path),
        schema_version=schema_version,
        frozen=frozen,
        taxonomy_assets=taxonomy_assets,
        model=model,
        data=data,
        loss=loss,
        sampler=sampler,
        output=output,
        transforms=transforms,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        reproducibility=reproducibility,
    )
