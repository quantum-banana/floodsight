from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from floodsight_data import __version__
from floodsight_data.acquisition import (
    import_archive,
    import_archives,
    import_directory,
    manual_acquisition_status,
)
from floodsight_data.common.atomic import atomic_write_json, atomic_write_text
from floodsight_data.common.materialize import MaterializationStrategy
from floodsight_data.common.segmentation_converter import convert_segmentation_dataset
from floodsight_data.config import PROJECT_ROOT
from floodsight_data.errors import DatasetToolError
from floodsight_data.hashing import IntegrityMode
from floodsight_data.manifests import read_json, validate_schema
from floodsight_data.paths import DATA_ROOT_ENV, DataPaths, resolve_data_root
from floodsight_data.registry import get_dataset, load_registry
from floodsight_data.reports import record_review, write_combined_report, write_dataset_report
from floodsight_data.taxonomy import (
    load_mapping,
    load_taxonomy,
    mapping_markdown,
    validate_mapping_targets,
)
from floodsight_data.validation import (
    guard_repository_artifacts,
    inspect_source,
    validate_dataset,
)
from floodsight_data.visdrone.converter import convert_visdrone_dataset
from floodsight_data.visualization import generate_inspection


def _add_data_root(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--data-root",
        required=required,
        help=f"External dataset root; defaults to {DATA_ROOT_ENV}.",
    )
    parser.add_argument("--cache-root", help="External download cache root.")


def _add_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=sorted(load_registry()))


def _add_conversion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--integrity", choices=[item.value for item in IntegrityMode], default="fast"
    )
    parser.add_argument(
        "--materialization",
        choices=[item.value for item in MaterializationStrategy],
        default="hardlink",
    )
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodsight-data",
        description=(
            "FloodSight Phase 3 dataset acquisition, validation, taxonomy, conversion, "
            "manifest, report, and inspection tools. No model training is included."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks for unexpected failures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check tooling and external-data configuration.")
    _add_data_root(doctor)

    registry = subparsers.add_parser("registry", help="Show the typed dataset registry.")
    registry.add_argument("--dataset", choices=sorted(load_registry()))

    taxonomy = subparsers.add_parser("taxonomy", help="Validate taxonomies and source mappings.")
    taxonomy.add_argument("--write-tables", action="store_true")
    taxonomy.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "docs" / "generated-mappings"
    )

    acquire = subparsers.add_parser("acquire", help="Show or run an official acquisition method.")
    _add_dataset(acquire)
    _add_data_root(acquire)

    archive = subparsers.add_parser("import-archive", help="Safely import a user-provided archive.")
    _add_dataset(archive)
    _add_data_root(archive)
    archive.add_argument("--archive", required=True, type=Path)
    archive.add_argument("--force", action="store_true")
    archive.add_argument("--dry-run", action="store_true")

    archives = subparsers.add_parser(
        "import-archives", help="Safely stage and atomically import multiple source archives."
    )
    _add_dataset(archives)
    _add_data_root(archives)
    archives.add_argument("--archive", required=True, action="append", type=Path)
    archives.add_argument("--force", action="store_true")
    archives.add_argument("--dry-run", action="store_true")

    directory = subparsers.add_parser(
        "import-directory", help="Import a user-provided extracted source directory."
    )
    _add_dataset(directory)
    _add_data_root(directory)
    directory.add_argument("--source", required=True, type=Path)
    directory.add_argument("--force", action="store_true")
    directory.add_argument("--dry-run", action="store_true")

    source = subparsers.add_parser(
        "inspect-source", help="Inventory source structure, labels, and pairing."
    )
    _add_dataset(source)
    _add_data_root(source)

    validate = subparsers.add_parser(
        "validate", help="Validate a source and audit duplicates/leakage."
    )
    _add_dataset(validate)
    _add_data_root(validate)

    convert = subparsers.add_parser(
        "convert", help="Convert one dataset without changing raw files."
    )
    _add_dataset(convert)
    _add_data_root(convert)
    _add_conversion_options(convert)

    manifest = subparsers.add_parser("manifest", help="Validate and summarize a prepared manifest.")
    _add_dataset(manifest)
    _add_data_root(manifest)

    inspect = subparsers.add_parser(
        "inspect", help="Generate deterministic annotation contact sheets."
    )
    _add_dataset(inspect)
    _add_data_root(inspect)
    inspect.add_argument("--split")
    inspect.add_argument("--count", type=int, default=24)
    inspect.add_argument("--seed", type=int, default=1337)

    review = subparsers.add_parser(
        "review", help="Record an explicit human review against the current fingerprint."
    )
    _add_dataset(review)
    _add_data_root(review)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--license-reviewed", action="store_true")
    review.add_argument("--mapping-reviewed", action="store_true")
    review.add_argument("--visual-reviewed", action="store_true")

    report = subparsers.add_parser("report", help="Generate JSON and Markdown health reports.")
    report.add_argument("--dataset", choices=sorted(load_registry()))
    report.add_argument("--all", action="store_true", help="Write the combined readiness report.")
    _add_data_root(report)

    fingerprint = subparsers.add_parser("fingerprint", help="Show a prepared dataset fingerprint.")
    _add_dataset(fingerprint)
    _add_data_root(fingerprint)

    prepare = subparsers.add_parser(
        "prepare", help="Validate, convert, and report one imported dataset."
    )
    _add_dataset(prepare)
    _add_data_root(prepare)
    _add_conversion_options(prepare)

    prepare_all = subparsers.add_parser(
        "prepare-all", help="Prepare all imported datasets without downloading them."
    )
    _add_data_root(prepare_all)
    _add_conversion_options(prepare_all)

    safeguard = subparsers.add_parser(
        "safeguard", help="Fail if datasets, archives, or model artifacts entered the repository."
    )
    safeguard.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    return parser


def _paths(args: argparse.Namespace, *, create: bool = False) -> DataPaths:
    paths = DataPaths.from_values(args.data_root, args.cache_root)
    if create:
        paths.ensure_layout(dry_run=bool(getattr(args, "dry_run", False)))
    return paths


def _doctor(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_data_root(args.data_root, required=False)
    datasets: dict[str, Any] = {}
    if data_root is not None:
        for dataset_id in load_registry():
            raw = data_root / "raw" / dataset_id
            datasets[dataset_id] = {
                "status": "PRESENT_NOT_VALIDATED" if raw.is_dir() else "MISSING",
                "path": str(raw),
            }
    else:
        datasets = {
            dataset_id: {"status": "DATA_ROOT_NOT_CONFIGURED"} for dataset_id in load_registry()
        }
    return {
        "tool": "floodsight-data",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_root": None if data_root is None else str(data_root),
        "data_root_configured": data_root is not None,
        "data_root_exists": bool(data_root and data_root.exists()),
        "datasets": datasets,
        "training_frameworks_installed_by_package": [],
        "status": "OK" if sys.version_info >= (3, 11) else "BLOCKED",
    }


def _registry(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry()
    if args.dataset:
        return registry[args.dataset].to_dict()
    return {dataset_id: record.to_dict() for dataset_id, record in registry.items()}


def _taxonomy(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy_files = (
        "product-taxonomy-v1.yaml",
        "segmentation-taxonomy-v1.yaml",
        "detection-taxonomy-v1.yaml",
    )
    taxonomies = {name: load_taxonomy(name)[0] for name in taxonomy_files}
    written: list[str] = []
    for dataset_id in load_registry():
        mapping = load_mapping(dataset_id)
        validate_mapping_targets(
            mapping,
            "detection-taxonomy-v1.yaml"
            if dataset_id == "visdrone_det"
            else "segmentation-taxonomy-v1.yaml",
        )
        if args.write_tables:
            path = args.output / f"{dataset_id}-mapping.md"
            atomic_write_text(path, mapping_markdown(mapping))
            written.append(str(path))
    return {"status": "VALID", "taxonomies": taxonomies, "mapping_tables": written}


def _convert(paths: DataPaths, dataset_id: str, args: argparse.Namespace) -> dict[str, Any]:
    integrity = IntegrityMode(args.integrity)
    materialization = MaterializationStrategy(args.materialization)
    if dataset_id == "visdrone_det":
        return convert_visdrone_dataset(
            paths,
            integrity=integrity,
            materialization=materialization,
            dry_run=args.dry_run,
        )
    return convert_segmentation_dataset(
        paths,
        dataset_id,
        integrity=integrity,
        materialization=materialization,
        dry_run=args.dry_run,
    )


def _prepared_manifest(paths: DataPaths, dataset_id: str) -> Path:
    version = "detection_v1" if dataset_id == "visdrone_det" else "segmentation_v1"
    return paths.manifests / f"{dataset_id}-{version}.json"


def _prepare(paths: DataPaths, dataset_id: str, args: argparse.Namespace) -> dict[str, Any]:
    validation = validate_dataset(paths, dataset_id)
    if not validation["valid"]:
        raise DatasetToolError(
            f"{dataset_id} validation has blocking errors.",
            code="validation_failed",
            details=[{"blocking_errors": validation["blocking_errors"]}],
            exit_code=3,
        )
    if not args.dry_run:
        atomic_write_json(paths.reports / f"{dataset_id}-source-inventory.json", validation)
    conversion = _convert(paths, dataset_id, args)
    if args.dry_run:
        report_result: dict[str, Any] = {"status": "SKIPPED_DRY_RUN"}
    else:
        json_path, markdown_path, report = write_dataset_report(paths, dataset_id)
        report_result = {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "ready_for_next_phase": report["ready_for_next_phase"],
        }
    return {"validation": validation, "conversion": conversion, "report": report_result}


def dispatch(args: argparse.Namespace) -> Any:
    command = args.command
    if command == "doctor":
        return _doctor(args)
    if command == "registry":
        return _registry(args)
    if command == "taxonomy":
        return _taxonomy(args)
    if command == "safeguard":
        violations = guard_repository_artifacts(args.repository_root.resolve())
        if violations:
            raise DatasetToolError(
                "Repository artifact safeguard failed.",
                code="repository_artifacts_detected",
                details=[{"paths": violations}],
                exit_code=4,
            )
        return {"status": "CLEAN", "repository_root": str(args.repository_root.resolve())}

    record = get_dataset(getattr(args, "dataset", "")) if getattr(args, "dataset", None) else None
    if command == "acquire":
        assert record is not None
        return manual_acquisition_status(record)
    if command == "import-archive":
        assert record is not None
        return import_archive(
            _paths(args, create=True),
            record,
            args.archive,
            force=args.force,
            dry_run=args.dry_run,
        )
    if command == "import-archives":
        assert record is not None
        return import_archives(
            _paths(args, create=True),
            record,
            args.archive,
            force=args.force,
            dry_run=args.dry_run,
        )
    if command == "import-directory":
        assert record is not None
        return import_directory(
            _paths(args, create=True),
            record,
            args.source,
            force=args.force,
            dry_run=args.dry_run,
        )
    if command == "inspect-source":
        assert record is not None
        return inspect_source(_paths(args), record.canonical_id)
    if command == "validate":
        assert record is not None
        return validate_dataset(_paths(args), record.canonical_id)
    if command == "convert":
        assert record is not None
        return _convert(_paths(args, create=True), record.canonical_id, args)
    if command == "manifest":
        assert record is not None
        path = _prepared_manifest(_paths(args), record.canonical_id)
        payload = read_json(path)
        validate_schema(payload, "dataset-manifest.schema.json")
        return {
            "status": "VALID",
            "path": str(path),
            "sample_count": len(payload["samples"]),
            "fingerprint": payload["fingerprint"],
        }
    if command == "inspect":
        assert record is not None
        return generate_inspection(
            _paths(args),
            record.canonical_id,
            split=args.split,
            count=args.count,
            seed=args.seed,
        )
    if command == "review":
        assert record is not None
        if not (args.license_reviewed or args.mapping_reviewed or args.visual_reviewed):
            raise DatasetToolError(
                "Select at least one explicit review attestation.",
                code="review_attestation_missing",
            )
        path = record_review(
            _paths(args),
            record.canonical_id,
            reviewer=args.reviewer,
            license_reviewed=args.license_reviewed,
            mapping_reviewed=args.mapping_reviewed,
            visual_reviewed=args.visual_reviewed,
        )
        return {"status": "RECORDED", "path": str(path)}
    if command == "report":
        paths = _paths(args, create=True)
        if args.all:
            json_path, markdown_path, report = write_combined_report(paths)
        elif args.dataset:
            json_path, markdown_path, report = write_dataset_report(paths, args.dataset)
        else:
            raise DatasetToolError("Choose --dataset or --all.", code="report_target_missing")
        return {"json": str(json_path), "markdown": str(markdown_path), "report": report}
    if command == "fingerprint":
        assert record is not None
        path = _prepared_manifest(_paths(args), record.canonical_id)
        manifest = read_json(path)
        return {
            "dataset_id": record.canonical_id,
            "integrity_mode": manifest["integrity_mode"],
            "fingerprint": manifest["fingerprint"],
        }
    if command == "prepare":
        assert record is not None
        return _prepare(_paths(args, create=True), record.canonical_id, args)
    if command == "prepare-all":
        paths = _paths(args, create=True)
        results: dict[str, Any] = {}
        failures = 0
        for dataset_id in load_registry():
            try:
                results[dataset_id] = _prepare(paths, dataset_id, args)
            except DatasetToolError as exc:
                failures += 1
                results[dataset_id] = {
                    "status": "BLOCKED",
                    "code": exc.code,
                    "message": exc.message,
                }
        if failures:
            raise DatasetToolError(
                f"{failures} dataset preparations are blocked.",
                code="prepare_all_incomplete",
                details=[{"datasets": results}],
                exit_code=3,
            )
        return {"status": "COMPLETE", "datasets": results}
    raise DatasetToolError(f"Unsupported command: {command}", code="command_invalid")


def _emit(payload: Any, *, as_json: bool, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=stream)
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            rendered = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
            print(f"{key}: {rendered}", file=stream)
    else:
        print(payload, file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except DatasetToolError as exc:
        _emit(
            {
                "status": "ERROR",
                "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            },
            as_json=args.json,
            error=True,
        )
        return exc.exit_code
    except Exception as exc:
        if args.debug:
            raise
        _emit(
            {
                "status": "ERROR",
                "error": {
                    "code": "unexpected_error",
                    "message": str(exc),
                    "details": [],
                },
            },
            as_json=args.json,
            error=True,
        )
        return 1
    _emit(result, as_json=args.json)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
