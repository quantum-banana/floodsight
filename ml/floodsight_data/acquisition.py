from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floodsight_data.common.archive import safe_extract_archive
from floodsight_data.common.atomic import atomic_write_json
from floodsight_data.errors import DatasetToolError
from floodsight_data.hashing import sha256_file, stable_digest
from floodsight_data.paths import DataPaths, ensure_contained
from floodsight_data.registry import DatasetRecord


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def manual_acquisition_status(record: DatasetRecord) -> dict[str, Any]:
    return {
        "dataset_id": record.canonical_id,
        "status": "MANUAL_ACTION_REQUIRED",
        "official_reference": record.official_reference,
        "license_review_state": record.license_review_state.value,
        "instructions": list(record.required_manual_steps),
        "supported_follow_up": ["import-archive", "import-directory"],
    }


def download_resumable(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download an explicitly requested stable URL with a `.part` resume file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if expected_sha256 is None or sha256_file(destination) == expected_sha256:
            return destination
        raise DatasetToolError(
            f"Existing download has an unexpected checksum: {destination}",
            code="download_checksum_mismatch",
        )
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "FloodSight-Data/0.3"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            status = getattr(response, "status", 200)
            if offset and status != 206:
                offset = 0
            mode = "ab" if offset else "wb"
            with partial.open(mode) as output:
                while chunk := response.read(chunk_size):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except (OSError, urllib.error.URLError) as exc:
        raise DatasetToolError(
            f"Download interrupted; partial data remains at {partial}: {exc}",
            code="download_failed",
        ) from exc
    if expected_sha256 and sha256_file(partial) != expected_sha256:
        raise DatasetToolError(
            f"Downloaded file failed SHA-256 verification: {partial}",
            code="download_checksum_mismatch",
        )
    os.replace(partial, destination)
    return destination


def _record_import(
    paths: DataPaths,
    record: DatasetRecord,
    *,
    source: Path,
    method: str,
    destination: Path,
) -> Path:
    metadata = {
        "dataset_id": record.canonical_id,
        "status": "IMPORTED_NOT_VALIDATED",
        "method": method,
        "source": str(source.resolve()),
        "source_size": source.stat().st_size if source.is_file() else None,
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "source_fingerprint": stable_digest(
            [
                item.relative_to(destination).as_posix()
                for item in sorted(destination.rglob("*"))
                if item.is_file()
            ]
        ),
        "destination": str(destination.resolve()),
        "imported_at": _utc_now(),
        "license_review_state": record.license_review_state.value,
    }
    metadata_path = paths.locks / f"{record.canonical_id}-acquisition.json"
    atomic_write_json(metadata_path, metadata)
    return metadata_path


def import_archive(
    paths: DataPaths,
    record: DatasetRecord,
    archive: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    destination = ensure_contained(paths.dataset_raw(record.canonical_id), paths.root)
    safe_extract_archive(archive.resolve(), destination, force=force, dry_run=dry_run)
    metadata_path = None
    if not dry_run:
        paths.locks.mkdir(parents=True, exist_ok=True)
        metadata_path = _record_import(
            paths,
            record,
            source=archive,
            method="user-archive",
            destination=destination,
        )
    return {
        "dataset_id": record.canonical_id,
        "status": "DRY_RUN" if dry_run else "IMPORTED_NOT_VALIDATED",
        "destination": str(destination),
        "metadata": None if metadata_path is None else str(metadata_path),
    }


def import_directory(
    paths: DataPaths,
    record: DatasetRecord,
    source: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise DatasetToolError(f"Source directory does not exist: {source}", code="source_missing")
    destination = ensure_contained(paths.dataset_raw(record.canonical_id), paths.root)
    if source == destination or destination in source.parents:
        raise DatasetToolError("Source and raw destination overlap.", code="unsafe_path")
    if destination.exists() and not force:
        raise DatasetToolError(
            f"Raw destination already exists: {destination}. Use --force explicitly to replace it.",
            code="destination_exists",
        )
    if not dry_run:
        staging = destination.parent / f".{destination.name}.importing"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        paths.locks.mkdir(parents=True, exist_ok=True)
        metadata_path = _record_import(
            paths,
            record,
            source=source,
            method="user-directory",
            destination=destination,
        )
    else:
        metadata_path = None
    return {
        "dataset_id": record.canonical_id,
        "status": "DRY_RUN" if dry_run else "IMPORTED_NOT_VALIDATED",
        "destination": str(destination),
        "metadata": None if metadata_path is None else str(metadata_path),
    }


def read_acquisition_metadata(paths: DataPaths, dataset_id: str) -> dict[str, Any] | None:
    path = paths.locks / f"{dataset_id}-acquisition.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else None
