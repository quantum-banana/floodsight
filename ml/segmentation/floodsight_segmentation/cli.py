"""Lazy-import-safe command line interface for SegFormer operations."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .approval import validate_human_approval
from .artifact import ModelArtifactSpec
from .config import load_config
from .errors import SegmentationError
from .guard import require_real_smoke_authorization, require_training_authorization
from .manifest import ManifestSpec

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "segformer_b2.yaml"


def _add_common_real_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="Torch device, for example cuda:0.")
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Explicitly unlock real manifest/model access after every gate passes.",
    )
    _add_model_artifact_arguments(parser)
    _add_approval_arguments(parser)


def _add_model_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-safetensors", type=Path, required=True)
    parser.add_argument("--model-safetensors-sha256", required=True)
    parser.add_argument("--model-provenance-record", type=Path, required=True)
    parser.add_argument("--model-provenance-record-sha256", required=True)


def _add_approval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approval-record", type=Path, required=True)
    parser.add_argument("--approval-record-sha256", required=True)


def _add_prepared_fast_path_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_source_transition: bool,
) -> None:
    """Add explicit, all-or-none runtime fast-path controls."""

    parser.add_argument("--prepared-fast-path-record", type=Path)
    parser.add_argument("--prepared-fast-path-record-sha256")
    parser.add_argument("--allow-prepared-fast-path", action="store_true")
    parser.add_argument("--loader-threads", type=int)
    parser.add_argument("--loader-prefetch-samples", type=int)
    parser.add_argument("--torch-cpu-threads", type=int)
    if allow_source_transition:
        parser.add_argument("--source-transition-record", type=Path)
        parser.add_argument("--source-transition-record-sha256")


def _add_manifest_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    label = prefix.replace("_", "-")
    parser.add_argument(
        f"--{label}-manifest",
        action="append",
        type=Path,
        required=True,
        help=f"Frozen {prefix} manifest; repeat once per source dataset.",
    )
    parser.add_argument(
        f"--{label}-manifest-sha256",
        action="append",
        required=True,
        help="Expected content hash, in the same order as the manifest arguments.",
    )
    parser.add_argument(
        f"--{label}-manifest-fingerprint",
        action="append",
        required=True,
        help="Expected canonical dataset fingerprint, in manifest argument order.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodsight-segmentation",
        description="Guarded FloodSight SegFormer-B2 training infrastructure.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke",
        help="Run an offline tiny-model smoke using generated tensors only.",
    )
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--output-dir", type=Path)

    real_smoke = commands.add_parser(
        "real-smoke",
        help="Run one optimizer step on a tiny Pool/source-specific frozen sample set.",
    )
    real_smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    real_smoke.add_argument("--data-root", type=Path, required=True)
    real_smoke.add_argument("--device", default="auto")
    real_smoke.add_argument("--manifest", action="append", type=Path, required=True)
    real_smoke.add_argument("--manifest-sha256", action="append", required=True)
    real_smoke.add_argument("--manifest-fingerprint", action="append", required=True)
    real_smoke.add_argument("--output-dir", type=Path, required=True)
    real_smoke.add_argument("--allow-real-smoke", action="store_true")
    _add_model_artifact_arguments(real_smoke)
    _add_prepared_fast_path_arguments(real_smoke, allow_source_transition=False)

    train = commands.add_parser("train", help="Run a future authorized real-data training job.")
    _add_common_real_arguments(train)
    _add_manifest_arguments(train, "train")
    _add_manifest_arguments(train, "validation")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    _add_prepared_fast_path_arguments(train, allow_source_transition=True)

    validate = commands.add_parser(
        "validate",
        help="Validate a real checkpoint against frozen validation manifests.",
    )
    _add_common_real_arguments(validate)
    _add_manifest_arguments(validate, "validation")
    validate.add_argument("--checkpoint", type=Path, required=True)
    return parser


def _specs(
    paths: list[Path],
    hashes: list[str],
    fingerprints: list[str],
    *,
    label: str,
) -> list[ManifestSpec]:
    if len(paths) != len(hashes) or len(paths) != len(fingerprints):
        raise SegmentationError(
            f"{label} manifest/SHA-256/fingerprint argument counts differ: "
            f"{len(paths)} != {len(hashes)} != {len(fingerprints)}."
        )
    return list(zip(paths, hashes, fingerprints, strict=True))


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _artifact_spec(args: argparse.Namespace) -> ModelArtifactSpec:
    return ModelArtifactSpec(
        safetensors_path=args.model_safetensors,
        safetensors_sha256=args.model_safetensors_sha256,
        provenance_path=args.model_provenance_record,
        provenance_sha256=args.model_provenance_record_sha256,
    )


def _prepared_fast_path_options(
    args: argparse.Namespace,
    *,
    allow_source_transition: bool,
) -> dict[str, Any]:
    """Return normalized fast-path CLI values only after an exact unlock."""

    values = {
        "prepared_fast_path_record": args.prepared_fast_path_record,
        "prepared_fast_path_record_sha256": args.prepared_fast_path_record_sha256,
        "loader_threads": args.loader_threads,
        "loader_prefetch_samples": args.loader_prefetch_samples,
        "torch_cpu_threads": args.torch_cpu_threads,
    }
    present = {key for key, value in values.items() if value is not None}
    if not present and not args.allow_prepared_fast_path:
        result: dict[str, Any] = {key: None for key in values}
        result["allow_prepared_fast_path"] = False
    elif present == set(values) and args.allow_prepared_fast_path is True:
        for key in ("loader_threads", "loader_prefetch_samples", "torch_cpu_threads"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SegmentationError(f"--{key.replace('_', '-')} must be positive.")
        result = {**values, "allow_prepared_fast_path": True}
    else:
        missing = sorted(set(values) - present)
        raise SegmentationError(
            "Prepared fast path requires its record, record SHA-256, loader thread count, "
            "prefetch-sample count, Torch CPU thread count, and "
            f"--allow-prepared-fast-path together; missing={missing}."
        )

    if allow_source_transition:
        transition_values = {
            "source_transition_record": args.source_transition_record,
            "source_transition_record_sha256": args.source_transition_record_sha256,
        }
        transition_present = {
            key for key, value in transition_values.items() if value is not None
        }
        if transition_present and transition_present != set(transition_values):
            raise SegmentationError(
                "Source transition record and record SHA-256 must be supplied together."
            )
        if transition_present and not args.allow_prepared_fast_path:
            raise SegmentationError(
                "A source transition is allowed only for the explicitly unlocked prepared "
                "fast path."
            )
        result.update(transition_values)
    return result


def _preauthorize(
    args: argparse.Namespace,
    *,
    operation: str,
    manifest_specs: list[ManifestSpec],
) -> None:
    """Reject absent/mismatched human approval before importing the ML runtime."""

    config = load_config(args.config)
    run_directory = (
        args.output_dir.expanduser().resolve()
        if operation == "TRAIN"
        else args.checkpoint.expanduser().resolve().parent
    )
    validate_human_approval(
        args.approval_record,
        expected_record_sha256=args.approval_record_sha256,
        operation=operation,
        config_sha256=config.sha256,
        manifest_specs=manifest_specs,
        taxonomy_sha256=config.taxonomy_assets.hashes,
        run_directory=run_directory,
        model_id=config.model.pretrained_model_name_or_path,
        model_revision=config.model.revision,
        model_artifact=_artifact_spec(args),
        real_smoke_root=config.output.real_smoke_root,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            from .smoke import run_synthetic_smoke

            if args.output_dir is not None:
                _print(run_synthetic_smoke(args.output_dir, device_name=args.device))
            else:
                with tempfile.TemporaryDirectory(prefix="floodsight-segformer-smoke-") as directory:
                    _print(run_synthetic_smoke(Path(directory), device_name=args.device))
            return 0

        if args.command == "real-smoke":
            require_real_smoke_authorization(args.allow_real_smoke)
            fast_path_options = _prepared_fast_path_options(
                args,
                allow_source_transition=False,
            )
            smoke_specs = _specs(
                args.manifest,
                args.manifest_sha256,
                args.manifest_fingerprint,
                label="smoke",
            )
            from .engine import run_real_manifest_smoke

            result = run_real_manifest_smoke(
                config_path=args.config,
                data_root=args.data_root,
                manifest_specs=smoke_specs,
                output_dir=args.output_dir,
                device_name=args.device,
                model_artifact_spec=_artifact_spec(args),
                allow_real_smoke=True,
                **fast_path_options,
            )
            _print(result)
            return 0

        # Refuse before importing engine/Torch or touching a manifest, checkpoint, or dataset.
        require_training_authorization(args.allow_training)
        if args.command == "train":
            fast_path_options = _prepared_fast_path_options(
                args,
                allow_source_transition=True,
            )
            train_specs = _specs(
                args.train_manifest,
                args.train_manifest_sha256,
                args.train_manifest_fingerprint,
                label="train",
            )
            validation_specs = _specs(
                args.validation_manifest,
                args.validation_manifest_sha256,
                args.validation_manifest_fingerprint,
                label="validation",
            )
            _preauthorize(
                args,
                operation="TRAIN",
                manifest_specs=[*train_specs, *validation_specs],
            )
            from .engine import run_training

            result = run_training(
                config_path=args.config,
                data_root=args.data_root,
                train_manifest_specs=train_specs,
                validation_manifest_specs=validation_specs,
                output_dir=args.output_dir,
                device_name=args.device,
                resume_checkpoint=args.resume,
                model_artifact_spec=_artifact_spec(args),
                approval_record=args.approval_record,
                approval_record_sha256=args.approval_record_sha256,
                allow_training=True,
                **fast_path_options,
            )
        else:
            validation_specs = _specs(
                args.validation_manifest,
                args.validation_manifest_sha256,
                args.validation_manifest_fingerprint,
                label="validation",
            )
            _preauthorize(args, operation="VALIDATE", manifest_specs=validation_specs)
            from .engine import run_validation

            result = run_validation(
                config_path=args.config,
                data_root=args.data_root,
                manifest_specs=validation_specs,
                checkpoint_path=args.checkpoint,
                device_name=args.device,
                model_artifact_spec=_artifact_spec(args),
                approval_record=args.approval_record,
                approval_record_sha256=args.approval_record_sha256,
                allow_training=True,
            )
        _print(result)
        return 0
    except (SegmentationError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
