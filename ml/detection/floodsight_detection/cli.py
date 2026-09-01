"""Command line entrypoints for validation, freezing, smoke, and guarded train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from floodsight_detection.config import PINNED_SEED, TrainingConfig, load_training_config
from floodsight_detection.contract import (
    DatasetContract,
    freeze_dataset_contract,
    validate_dataset_contract,
)
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.runtime import (
    bound_real_smoke_output_directory,
    bound_training_device,
    bound_training_output_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floodsight-detection")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "freeze"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        if name == "freeze":
            command.add_argument("--output", type=Path, required=True)
    train = subcommands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--run-name", required=True)
    train.add_argument("--device")
    train.add_argument("--resume", type=Path)
    train.add_argument("--weights-audit", type=Path)
    train.add_argument("--approval", type=Path)
    train.add_argument("--real-smoke-report", type=Path)
    train.add_argument("--allow-training", action="store_true")
    smoke = subcommands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--seed", type=int, default=20260831)
    smoke.add_argument("--allow-synthetic-smoke", action="store_true")
    real_smoke = subcommands.add_parser("real-smoke")
    real_smoke.add_argument("--config", type=Path, required=True)
    real_smoke.add_argument("--manifest", type=Path, required=True)
    real_smoke.add_argument("--data-root", type=Path, required=True)
    real_smoke.add_argument("--weights-audit", type=Path)
    real_smoke.add_argument("--output", type=Path, required=True)
    real_smoke.add_argument("--device", required=True)
    real_smoke.add_argument("--allow-real-smoke", action="store_true")
    return parser


def _contract(config: TrainingConfig, manifest: Path, data_root: Path) -> DatasetContract:
    gate = config.dataset
    return validate_dataset_contract(
        manifest,
        data_root,
        expected_manifest_path=gate.manifest_path,
        expected_manifest_sha256=gate.manifest_sha256,
        expected_dataset_fingerprint=gate.dataset_fingerprint,
        expected_source_version=gate.source_version,
        verify_image_hashes=gate.verify_image_hashes,
        require_full_integrity=gate.require_full_integrity,
        required_splits=gate.required_splits,
        require_all_train_classes=gate.require_all_train_classes,
        reject_duplicate_images=gate.reject_duplicate_images,
    )


def run(arguments: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(arguments)
    if args.command == "smoke":
        if not args.allow_synthetic_smoke:
            raise DetectionInfrastructureError(
                "Synthetic smoke is disabled; pass --allow-synthetic-smoke explicitly.",
                code="synthetic_smoke_not_authorized",
            )
        from floodsight_detection.smoke import run_synthetic_smoke

        return run_synthetic_smoke(
            args.output,
            allow_synthetic_smoke=True,
            device=args.device,
            seed=args.seed,
        )
    if args.command == "real-smoke" and not args.allow_real_smoke:
        raise DetectionInfrastructureError(
            "Real-manifest smoke is disabled; pass --allow-real-smoke explicitly.",
            code="real_smoke_not_authorized",
        )
    if args.command == "real-smoke" and args.weights_audit is None:
        raise DetectionInfrastructureError(
            "Real-manifest smoke requires an audited local --weights-audit record.",
            code="real_smoke_not_authorized",
        )
    if args.command == "train" and not args.allow_training:
        # Fail before reading data/configuration or importing either ML framework.
        raise DetectionInfrastructureError(
            "Real detection training is disabled; pass --allow-training explicitly.",
            code="training_not_authorized",
        )
    if args.command == "train" and (
        args.weights_audit is None
        or args.approval is None
        or args.real_smoke_report is None
    ):
        raise DetectionInfrastructureError(
            "Training requires --weights-audit, --real-smoke-report, and --approval "
            "in addition to --allow-training.",
            code="training_not_authorized",
        )
    if args.command in {"train", "real-smoke"}:
        # This must run before configuration or dataset access.  Mutating the
        # environment after interpreter startup cannot fix Python hash RNG.
        from floodsight_detection.determinism import require_prestarted_hash_seed

        require_prestarted_hash_seed(PINNED_SEED)
    config = load_training_config(args.config)
    selected_device = (
        bound_training_device(str(config.train["device"]), args.device)
        if args.command in {"train", "real-smoke"}
        else None
    )
    training_output_root = (
        bound_training_output_root(config.output.run_root, args.output_root)
        if args.command == "train"
        else None
    )
    real_smoke_output = (
        bound_real_smoke_output_directory(config.output.real_smoke_root, args.output)
        if args.command == "real-smoke"
        else None
    )
    if args.command == "real-smoke":
        # Fail before manifest access or output creation when the explicitly
        # authorized smoke is not in the audited offline H100 environment.
        from floodsight_detection.runtime import validate_full_training_runtime

        validate_full_training_runtime(str(selected_device))
    contract = _contract(config, args.manifest, args.data_root)
    if args.command == "real-smoke":
        from floodsight_detection.real_smoke import run_real_manifest_smoke
        from floodsight_detection.weights import load_weight_audit

        weights = load_weight_audit(
            args.weights_audit,
            expected_filename=config.model,
            expected_weight_path=config.model_path,
            expected_weight_sha256=config.model_sha256,
            expected_audit_path=config.weight_audit_path,
            expected_audit_sha256=config.weight_audit_sha256,
            require_license_approval=False,
        )
        return run_real_manifest_smoke(
            config,
            contract,
            weights,
            real_smoke_output,
            allow_real_smoke=True,
            device=str(selected_device),
        )
    if args.command == "validate":
        return {"status": "PASS", "config_sha256": config.sha256, **contract.summary()}
    if args.command == "freeze":
        yaml_path = freeze_dataset_contract(contract, args.output)
        return {
            "status": "PASS",
            "config_sha256": config.sha256,
            "data_yaml": str(yaml_path),
            **contract.summary(),
        }
    from floodsight_detection.approval import load_training_approval
    from floodsight_detection.real_smoke import load_real_smoke_attestation
    from floodsight_detection.ultralytics_runtime import execute_training
    from floodsight_detection.weights import (
        load_weight_audit,
        validate_training_license_disposition,
    )

    weights = load_weight_audit(
        args.weights_audit,
        expected_filename=config.model,
        expected_weight_path=config.model_path,
        expected_weight_sha256=config.model_sha256,
        expected_audit_path=config.weight_audit_path,
        expected_audit_sha256=config.weight_audit_sha256,
        require_license_approval=False,
    )
    real_smoke = load_real_smoke_attestation(
        args.real_smoke_report,
        config=config,
        contract=contract,
        weights=weights,
    )
    approval = load_training_approval(
        args.approval,
        config_sha256=config.sha256,
        manifest_sha256=contract.manifest_sha256,
        dataset_fingerprint=contract.dataset_fingerprint,
        weights_sha256=weights.sha256,
        weights_path=weights.path,
        weight_audit_path=weights.audit_path,
        weight_audit_sha256=weights.audit_sha256,
        run_name=args.run_name,
        output_root=training_output_root,
        device=str(selected_device),
        manifest_id=config.dataset.manifest_id,
        dataset_id=config.dataset.dataset_id,
        preparation_version=config.dataset.preparation_version,
        taxonomy_version=config.dataset.taxonomy_version,
        taxonomy_sha256=config.dataset.taxonomy_sha256,
        mapping_version=config.dataset.mapping_version,
        mapping_sha256=config.dataset.mapping_sha256,
        real_smoke_report_path=real_smoke.path,
        real_smoke_report_sha256=real_smoke.sha256,
    )
    validate_training_license_disposition(
        weights,
        review_disposition=approval.review_disposition,
    )

    return execute_training(
        config,
        contract,
        output_root=training_output_root,
        run_name=args.run_name,
        allow_training=True,
        weights=weights,
        approval=approval,
        real_smoke=real_smoke,
        resume_checkpoint=args.resume,
        device_override=args.device,
    )


def entrypoint(arguments: list[str] | None = None) -> int:
    try:
        result = run(arguments)
    except DetectionInfrastructureError as exc:
        print(json.dumps({"status": "FAIL", "error": exc.to_dict()}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
