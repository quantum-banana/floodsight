"""Strict loading of the frozen FloodSight YOLO training configuration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floodsight_detection.contract import DETECTION_CLASSES
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file

_EXACT_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:\+[A-Za-z0-9.]+)?$")
_INTEGER_DEVICE = re.compile(r"^(0|[1-9][0-9]*)$")

PINNED_MODEL = "yolo11l.pt"
PINNED_MODEL_PATH = Path(
    "/data/floodsight-workspace/floodsight-cache/ml/models/yolo11l/v8.3.0/yolo11l.pt"
)
PINNED_MODEL_SHA256 = "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"
PINNED_WEIGHT_AUDIT_PATH = Path(
    "/data/floodsight-workspace/floodsight-cache/ml/models/yolo11l/v8.3.0/"
    "yolo11l-weight-audit.json"
)
PINNED_WEIGHT_AUDIT_SHA256 = (
    "7dda1c10d0164bed756d2072442727eaaccbacacc9fd792e4c6c20cf4ef8af63"
)
PINNED_ULTRALYTICS_VERSION = "8.3.222"
PINNED_TORCH_VERSION = "2.13.0+cu130"
PINNED_TORCHVISION_VERSION = "0.28.0+cu130"
PINNED_SEED = 20260831
DETECTION_RUN_ROOT = Path(
    "/data/floodsight-workspace/floodsight-datasets/runs/detection"
)
DETECTION_REAL_SMOKE_ROOT = Path(
    "/data/floodsight-workspace/floodsight-datasets/runs/detection-real-smoke"
)
EXPECTED_MANIFEST_ID = "visdrone_det-detection_v2"
EXPECTED_MANIFEST_PATH = Path(
    "/data/floodsight-workspace/floodsight-datasets/manifests/"
    "visdrone_det-detection_v2.json"
)
EXPECTED_MANIFEST_SHA256 = (
    "87653f6c908d7957f7eb286c5e957a5dcaa004e2adcfa1e12ed37e9932bde6df"
)
EXPECTED_DATASET_ID = "visdrone_det"
EXPECTED_DATASET_FINGERPRINT = (
    "6e6c9c1b940b59b61f1f39348d97893417371e87b26aed0c6d1dcd45815efbf4"
)
EXPECTED_SOURCE_VERSION = "VisDrone2019-DET-official-train-val-sha256-locked"
EXPECTED_PREPARATION_VERSION = "detection_v2"
EXPECTED_TAXONOMY_VERSION = "detection-taxonomy-v1"
EXPECTED_TAXONOMY_SHA256 = "4000f1bb75b2e4687d60027a72dd3f428c9bed8ba8de918c0892e3f115bdf535"
EXPECTED_MAPPING_VERSION = "visdrone-mapping-v1"
EXPECTED_MAPPING_SHA256 = "ad0d67626195744bfc908cfc64c36b15a046d53f35a466791101256aa8681ad8"

# Every Ultralytics 8.3.222 DEFAULT_CFG_DICT key is deliberately classified.
# Train/validation keys must be present in the frozen config, runtime keys are
# supplied by this package, and the remaining keys are inapplicable to a detect
# train/validation run.  A runtime/test regression checks this partition against
# the installed pinned Ultralytics distribution so a new hidden default fails
# closed instead of silently entering a run.
ULTRALYTICS_RUNTIME_KEYS = frozenset(
    {"task", "mode", "model", "data", "project", "name", "exist_ok", "resume"}
)
ULTRALYTICS_INAPPLICABLE_KEYS = frozenset(
    {
        # Segmentation, pose, and classification-only settings.
        "overlap_mask",
        "mask_ratio",
        "dropout",
        "pose",
        "kobj",
        "auto_augment",
        "erasing",
        # Prediction, display, export, and tracking settings.
        "source",
        "vid_stride",
        "stream_buffer",
        "retina_masks",
        "embed",
        "show",
        "save_frames",
        "show_labels",
        "show_conf",
        "show_boxes",
        "line_width",
        "format",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "opset",
        "workspace",
        "nms",
        "cfg",
        "tracker",
    }
)
ULTRALYTICS_TRAIN_VAL_KEYS = frozenset(
    {
        "epochs",
        "time",
        "imgsz",
        "batch",
        "workers",
        "device",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "close_mosaic",
        "amp",
        "cache",
        "rect",
        "multi_scale",
        "patience",
        "save",
        "save_period",
        "plots",
        "val",
        "split",
        "save_json",
        "conf",
        "iou",
        "max_det",
        "half",
        "dnn",
        "augment",
        "visualize",
        "agnostic_nms",
        "classes",
        "save_txt",
        "save_conf",
        "save_crop",
        "pretrained",
        "verbose",
        "deterministic",
        "seed",
        "single_cls",
        "fraction",
        "profile",
        "freeze",
        "compile",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "copy_paste_mode",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "bgr",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "dfl",
        "nbs",
    }
)
_TRAIN_KEYS = ULTRALYTICS_TRAIN_VAL_KEYS


def validate_ultralytics_default_contract(defaults: dict[str, Any]) -> None:
    """Reject any unclassified Ultralytics default or a stale classification."""

    expected = (
        ULTRALYTICS_TRAIN_VAL_KEYS
        | ULTRALYTICS_RUNTIME_KEYS
        | ULTRALYTICS_INAPPLICABLE_KEYS
    )
    observed = frozenset(defaults)
    if observed != expected:
        _fail(
            "Ultralytics DEFAULT_CFG_DICT drifted from the audited 8.3.222 key contract.",
            "ultralytics_default_contract_mismatch",
        )


@dataclass(frozen=True, slots=True)
class DatasetGateConfig:
    manifest_path: Path
    manifest_sha256: str
    manifest_id: str
    dataset_id: str
    dataset_fingerprint: str
    source_version: str
    preparation_version: str
    taxonomy_version: str
    taxonomy_sha256: str
    mapping_version: str
    mapping_sha256: str
    require_full_integrity: bool
    verify_image_hashes: bool
    reject_duplicate_images: bool
    require_all_train_classes: bool
    required_splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputConfig:
    run_root: Path
    real_smoke_root: Path
    new_run_policy: str
    resume_policy: str
    last_checkpoint_filename: str
    best_checkpoint_filename: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    path: Path
    sha256: str
    model: str
    model_path: Path
    model_sha256: str
    weight_audit_path: Path
    weight_audit_sha256: str
    ultralytics_version: str
    torch_version: str
    torchvision_version: str
    classes: dict[int, str]
    dataset: DatasetGateConfig
    output: OutputConfig
    train: dict[str, Any]


def _fail(message: str, code: str = "training_config_invalid") -> None:
    raise DetectionInfrastructureError(message, code=code)


def _object(payload: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        _fail(f"Training configuration field {field!r} must be an object.")
    return payload


def _only_keys(payload: dict[str, Any], allowed: set[str] | frozenset[str], *, field: str) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        _fail(f"Unexpected {field} keys: {', '.join(unexpected)}.")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DetectionInfrastructureError(
            f"Unable to read training configuration: {path}",
            code="training_config_unreadable",
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # PyYAML belongs to the pinned ML environment, not this import-safe core.
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DetectionInfrastructureError(
                "Configuration is not JSON-compatible YAML and PyYAML could not parse it.",
                code="training_config_invalid",
            ) from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise DetectionInfrastructureError(
                "Configuration contains invalid YAML.",
                code="training_config_invalid",
            ) from exc
    if not isinstance(payload, dict):
        _fail("Training configuration must contain an object.")
    return payload


def version_matches(version: str, required: str) -> bool:
    """Require exact distribution versions, including CUDA local-version tags."""

    if _EXACT_VERSION.fullmatch(required) is None:
        _fail(f"Invalid exact dependency version {required!r}.", "dependency_version_invalid")
    return version == required


def _require_identity_file(relative_path: str, expected_sha256: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / relative_path
    if path.is_symlink() or not path.is_file():
        _fail(f"Frozen detection identity file is missing or unsafe: {relative_path}.")
    if sha256_file(path) != expected_sha256:
        _fail(f"Frozen detection identity file hash drifted: {relative_path}.")


def _require_exact_external_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        _fail(f"Frozen {label} is missing or unsafe: {path}.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DetectionInfrastructureError(
            f"Frozen {label} cannot be resolved: {path}.", code="training_config_invalid"
        ) from exc
    if resolved != path or sha256_file(path) != expected_sha256:
        _fail(f"Frozen {label} path or SHA-256 drifted: {path}.")


def validate_frozen_model_artifacts(config: TrainingConfig) -> dict[str, str]:
    """Re-read the exact config-bound model and provenance audit identities."""

    if (
        config.model != PINNED_MODEL
        or config.model_path != PINNED_MODEL_PATH
        or config.model_sha256 != PINNED_MODEL_SHA256
        or config.weight_audit_path != PINNED_WEIGHT_AUDIT_PATH
        or config.weight_audit_sha256 != PINNED_WEIGHT_AUDIT_SHA256
    ):
        _fail("Frozen detector model identity differs from the canonical Phase-K artifact.")
    _require_exact_external_file(
        config.model_path, config.model_sha256, label="detector weight artifact"
    )
    _require_exact_external_file(
        config.weight_audit_path,
        config.weight_audit_sha256,
        label="detector weight audit",
    )
    return {
        "weights_path": str(config.model_path),
        "weights_sha256": config.model_sha256,
        "weight_audit_path": str(config.weight_audit_path),
        "weight_audit_sha256": config.weight_audit_sha256,
    }


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    payload = _read_yaml_or_json(config_path)
    _only_keys(
        payload,
        {
            "schema_version",
            "task",
            "model",
            "classes",
            "dataset_gate",
            "output",
            "train",
        },
        field="top-level",
    )
    if payload.get("schema_version") != "floodsight-yolo-training-config-v4":
        _fail("Unsupported detection training configuration schema.")
    if payload.get("task") != "detect":
        _fail("YOLO training task must be 'detect'.")
    model = _object(payload.get("model"), field="model")
    _only_keys(
        model,
        {
            "weights_filename",
            "weights_path",
            "weights_sha256",
            "weight_audit_path",
            "weight_audit_sha256",
            "ultralytics_version",
            "torch_version",
            "torchvision_version",
        },
        field="model",
    )
    weights = model.get("weights_filename")
    version_spec = model.get("ultralytics_version")
    torch_version = model.get("torch_version")
    torchvision_version = model.get("torchvision_version")
    if not isinstance(weights, str) or not weights or "/" in weights or "\\" in weights:
        _fail("Model weights_filename must be the audited local artifact basename.")
    if not isinstance(version_spec, str) or _EXACT_VERSION.fullmatch(version_spec) is None:
        _fail("Ultralytics version must be an exact X.Y.Z version.")
    if not isinstance(torch_version, str) or _EXACT_VERSION.fullmatch(torch_version) is None:
        _fail("Torch version must be an exact X.Y.Z[+build] version.")
    if (
        not isinstance(torchvision_version, str)
        or _EXACT_VERSION.fullmatch(torchvision_version) is None
    ):
        _fail("Torchvision version must be an exact X.Y.Z[+build] version.")
    pinned_model = {
        "weights_filename": PINNED_MODEL,
        "weights_path": str(PINNED_MODEL_PATH),
        "weights_sha256": PINNED_MODEL_SHA256,
        "weight_audit_path": str(PINNED_WEIGHT_AUDIT_PATH),
        "weight_audit_sha256": PINNED_WEIGHT_AUDIT_SHA256,
        "ultralytics_version": PINNED_ULTRALYTICS_VERSION,
        "torch_version": PINNED_TORCH_VERSION,
        "torchvision_version": PINNED_TORCHVISION_VERSION,
    }
    if model != pinned_model:
        _fail("Model and dependency versions must match the frozen detector stack exactly.")
    _require_exact_external_file(
        PINNED_MODEL_PATH, PINNED_MODEL_SHA256, label="detector weight artifact"
    )
    _require_exact_external_file(
        PINNED_WEIGHT_AUDIT_PATH,
        PINNED_WEIGHT_AUDIT_SHA256,
        label="detector weight audit",
    )

    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, dict):
        _fail("classes must be an ID-to-name object.")
    try:
        classes = {int(key): str(value) for key, value in raw_classes.items()}
    except (TypeError, ValueError) as exc:
        raise DetectionInfrastructureError(
            "classes contains a non-integer ID.", code="training_config_invalid"
        ) from exc
    if classes != DETECTION_CLASSES:
        _fail("Configured detector classes do not match detection-taxonomy-v1.")

    dataset = _object(payload.get("dataset_gate"), field="dataset_gate")
    _only_keys(
        dataset,
        {
            "manifest_path",
            "manifest_sha256",
            "manifest_id",
            "dataset_id",
            "dataset_fingerprint",
            "source_version",
            "preparation_version",
            "taxonomy_version",
            "taxonomy_sha256",
            "mapping_version",
            "mapping_sha256",
            "require_full_integrity",
            "verify_image_hashes",
            "reject_duplicate_images",
            "require_all_train_classes",
            "required_splits",
        },
        field="dataset_gate",
    )
    identity = {
        "manifest_path": str(EXPECTED_MANIFEST_PATH),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_id": EXPECTED_MANIFEST_ID,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "source_version": EXPECTED_SOURCE_VERSION,
        "preparation_version": EXPECTED_PREPARATION_VERSION,
        "taxonomy_version": EXPECTED_TAXONOMY_VERSION,
        "taxonomy_sha256": EXPECTED_TAXONOMY_SHA256,
        "mapping_version": EXPECTED_MAPPING_VERSION,
        "mapping_sha256": EXPECTED_MAPPING_SHA256,
    }
    if any(dataset.get(key) != value for key, value in identity.items()):
        _fail("Detection manifest, taxonomy, or mapping identity drifted.")
    _require_identity_file(
        "shared/taxonomy/detection-taxonomy-v1.yaml", EXPECTED_TAXONOMY_SHA256
    )
    _require_identity_file("shared/taxonomy/visdrone-mapping-v1.yaml", EXPECTED_MAPPING_SHA256)
    _require_exact_external_file(
        EXPECTED_MANIFEST_PATH,
        EXPECTED_MANIFEST_SHA256,
        label="VisDrone detection_v2 manifest",
    )
    boolean_keys = (
        "require_full_integrity",
        "verify_image_hashes",
        "reject_duplicate_images",
        "require_all_train_classes",
    )
    if any(not isinstance(dataset.get(key), bool) for key in boolean_keys):
        _fail("All dataset gate switches must be explicit booleans.")
    if any(dataset[key] is not True for key in boolean_keys):
        _fail("All full-training dataset integrity gates must remain enabled.")
    splits = dataset.get("required_splits")
    if (
        not isinstance(splits, list)
        or not splits
        or any(not isinstance(item, str) for item in splits)
        or len(splits) != len(set(splits))
        or splits != ["train", "val"]
    ):
        _fail("required_splits must be exactly ['train', 'val'].")

    output = _object(payload.get("output"), field="output")
    expected_output = {
        "run_root": str(DETECTION_RUN_ROOT),
        "real_smoke_root": str(DETECTION_REAL_SMOKE_ROOT),
        "new_run_policy": "exclusive_direct_child",
        "resume_policy": "approved_run_last_only",
        "last_checkpoint_filename": "last.pt",
        "best_checkpoint_filename": "best.pt",
    }
    _only_keys(output, set(expected_output), field="output")
    if output != expected_output:
        _fail("The frozen detection output/run/resume policy drifted.")

    train = _object(payload.get("train"), field="train")
    _only_keys(train, _TRAIN_KEYS, field="train")
    missing = sorted(_TRAIN_KEYS - train.keys())
    if missing:
        _fail(f"Training configuration is missing: {', '.join(missing)}.")
    positive_integers = (
        "epochs",
        "imgsz",
        "batch",
        "close_mosaic",
        "patience",
        "save_period",
        "max_det",
        "nbs",
    )
    if any(
        isinstance(train[key], bool) or not isinstance(train[key], int) or train[key] < 1
        for key in positive_integers
    ):
        _fail("epochs, imgsz, and batch must be positive integers.")
    if (
        isinstance(train["workers"], bool)
        or not isinstance(train["workers"], int)
        or train["workers"] < 0
    ):
        _fail("workers must be a non-negative integer.")
    if train["workers"] != 0:
        _fail("workers must remain 0 for the host's 64 MiB shared-memory limit.")
    if isinstance(train["seed"], bool) or not isinstance(train["seed"], int) or train["seed"] < 0:
        _fail("seed must be a non-negative integer.")
    if train["seed"] != PINNED_SEED:
        _fail(f"The frozen detector seed must remain {PINNED_SEED}.")
    if train["imgsz"] % 32 != 0:
        _fail("imgsz must be divisible by the detector's 32-pixel maximum stride.")
    if train["close_mosaic"] > train["epochs"] or train["save_period"] > train["epochs"]:
        _fail("close_mosaic and save_period must not exceed epochs.")
    if train["optimizer"] != "AdamW":
        _fail("The frozen detector optimizer must be AdamW.")
    if not isinstance(train["device"], str) or _INTEGER_DEVICE.fullmatch(train["device"]) is None:
        _fail("The frozen detector device must be one explicit non-negative GPU index.")
    boolean_train_keys = (
        "amp",
        "cache",
        "cos_lr",
        "deterministic",
        "multi_scale",
        "plots",
        "pretrained",
        "rect",
        "save",
        "val",
        "verbose",
        "single_cls",
        "profile",
        "compile",
        "save_json",
        "half",
        "dnn",
        "augment",
        "visualize",
        "agnostic_nms",
        "save_txt",
        "save_conf",
        "save_crop",
    )
    if any(not isinstance(train[key], bool) for key in boolean_train_keys):
        _fail("All frozen detector switches must be explicit booleans.")
    expected_switches = {
        "amp": True,
        "cache": False,
        "cos_lr": True,
        "deterministic": True,
        "multi_scale": True,
        "plots": True,
        "pretrained": True,
        "rect": False,
        "save": True,
        "val": True,
        "verbose": True,
        "single_cls": False,
        "profile": False,
        "compile": False,
        "save_json": False,
        "half": False,
        "dnn": False,
        "augment": False,
        "visualize": False,
        "agnostic_nms": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
    }
    switch_drift = {
        key: {"expected": expected, "actual": train[key]}
        for key, expected in expected_switches.items()
        if train[key] is not expected
    }
    if switch_drift:
        _fail("One or more frozen detector boolean switches drifted.")
    if train["time"] is not None:
        _fail("time must remain null so the epoch envelope cannot be replaced.")
    if train["freeze"] is not None or train["classes"] is not None:
        _fail("freeze and classes must remain null for full-model, all-class training.")
    if train["split"] != "val":
        _fail("The frozen validation split must be 'val'.")
    if train["fraction"] != 1.0:
        _fail("fraction must remain 1.0 so the full sanitized training split is used.")
    if train["copy_paste_mode"] not in {"flip", "mixup"}:
        _fail("copy_paste_mode must be 'flip' or 'mixup'.")
    numeric_ranges = {
        "lr0": (0.0, 1.0, False, False),
        "lrf": (0.0, 1.0, False, True),
        "momentum": (0.0, 1.0, True, False),
        "weight_decay": (0.0, 1.0, True, True),
        "warmup_epochs": (0.0, float(train["epochs"]), True, True),
        "warmup_momentum": (0.0, 1.0, True, False),
        "warmup_bias_lr": (0.0, 1.0, True, True),
        "box": (0.0, 100.0, False, True),
        "cls": (0.0, 100.0, False, True),
        "dfl": (0.0, 100.0, False, True),
        "fraction": (0.0, 1.0, False, True),
        "iou": (0.0, 1.0, True, True),
        "mosaic": (0.0, 1.0, True, True),
        "mixup": (0.0, 1.0, True, True),
        "cutmix": (0.0, 1.0, True, True),
        "copy_paste": (0.0, 1.0, True, True),
        "degrees": (0.0, 180.0, True, True),
        "translate": (0.0, 1.0, True, True),
        "scale": (0.0, 1.0, True, True),
        "shear": (0.0, 180.0, True, True),
        "perspective": (0.0, 0.001, True, True),
        "fliplr": (0.0, 1.0, True, True),
        "flipud": (0.0, 1.0, True, True),
        "hsv_h": (0.0, 1.0, True, True),
        "hsv_s": (0.0, 1.0, True, True),
        "hsv_v": (0.0, 1.0, True, True),
        "bgr": (0.0, 1.0, True, True),
    }
    for key, (minimum, maximum, include_minimum, include_maximum) in numeric_ranges.items():
        value = train[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{key} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            _fail(f"{key} must be finite.")
        minimum_ok = numeric >= minimum if include_minimum else numeric > minimum
        maximum_ok = numeric <= maximum if include_maximum else numeric < maximum
        if not minimum_ok or not maximum_ok:
            left = "[" if include_minimum else "("
            right = "]" if include_maximum else ")"
            _fail(f"{key} must be in {left}{minimum},{maximum}{right}.")
    confidence = train["conf"]
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        _fail("conf must be null or a finite value in [0,1].")

    return TrainingConfig(
        path=config_path,
        sha256=sha256_file(config_path),
        model=weights,
        model_path=PINNED_MODEL_PATH,
        model_sha256=PINNED_MODEL_SHA256,
        weight_audit_path=PINNED_WEIGHT_AUDIT_PATH,
        weight_audit_sha256=PINNED_WEIGHT_AUDIT_SHA256,
        ultralytics_version=version_spec,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        classes=classes,
        dataset=DatasetGateConfig(
            manifest_path=EXPECTED_MANIFEST_PATH,
            manifest_sha256=dataset["manifest_sha256"],
            manifest_id=dataset["manifest_id"],
            dataset_id=dataset["dataset_id"],
            dataset_fingerprint=dataset["dataset_fingerprint"],
            source_version=dataset["source_version"],
            preparation_version=dataset["preparation_version"],
            taxonomy_version=dataset["taxonomy_version"],
            taxonomy_sha256=dataset["taxonomy_sha256"],
            mapping_version=dataset["mapping_version"],
            mapping_sha256=dataset["mapping_sha256"],
            require_full_integrity=dataset["require_full_integrity"],
            verify_image_hashes=dataset["verify_image_hashes"],
            reject_duplicate_images=dataset["reject_duplicate_images"],
            require_all_train_classes=dataset["require_all_train_classes"],
            required_splits=tuple(splits),
        ),
        output=OutputConfig(
            run_root=DETECTION_RUN_ROOT,
            real_smoke_root=DETECTION_REAL_SMOKE_ROOT,
            new_run_policy=output["new_run_policy"],
            resume_policy=output["resume_policy"],
            last_checkpoint_filename=output["last_checkpoint_filename"],
            best_checkpoint_filename=output["best_checkpoint_filename"],
        ),
        train=dict(train),
    )
