from __future__ import annotations

import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from floodsight_data.errors import DatasetToolError

MAX_ARCHIVE_MEMBERS = 2_000_000
MAX_EXPANDED_BYTES = 200 * 1024 * 1024 * 1024


def _member_path(name: str, destination: Path) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise DatasetToolError(
            f"Archive contains an unsafe path: {name}",
            code="archive_path_traversal",
        )
    if pure.parts and ":" in pure.parts[0]:
        raise DatasetToolError(
            f"Archive contains an absolute Windows path: {name}",
            code="archive_path_traversal",
        )
    target = (destination / Path(*pure.parts)).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise DatasetToolError(
            f"Archive path escapes the destination: {name}",
            code="archive_path_traversal",
        ) from exc
    return target


def _validate_limits(member_count: int, expanded_bytes: int) -> None:
    if member_count > MAX_ARCHIVE_MEMBERS or expanded_bytes > MAX_EXPANDED_BYTES:
        raise DatasetToolError(
            "Archive exceeds the configured extraction safety limits.",
            code="archive_too_large",
        )


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        _validate_limits(len(members), sum(member.file_size for member in members))
        for member in members:
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DatasetToolError(
                    f"Archive symbolic links are not allowed: {member.filename}",
                    code="archive_unsafe_member",
                )
            target = _member_path(member.filename, destination)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:*") as bundle:
        members = bundle.getmembers()
        _validate_limits(len(members), sum(member.size for member in members if member.isfile()))
        for member in members:
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise DatasetToolError(
                    f"Archive links and device entries are not allowed: {member.name}",
                    code="archive_unsafe_member",
                )
            target = _member_path(member.name, destination)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise DatasetToolError(
                    f"Unsupported archive member: {member.name}",
                    code="archive_unsafe_member",
                )
            source = bundle.extractfile(member)
            if source is None:
                raise DatasetToolError(
                    f"Unable to read archive member: {member.name}",
                    code="archive_invalid",
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def validate_archive_type(archive: Path) -> str:
    if not archive.is_file():
        raise DatasetToolError(f"Archive does not exist: {archive}", code="archive_missing")
    if zipfile.is_zipfile(archive):
        return "zip"
    if tarfile.is_tarfile(archive):
        return "tar"
    raise DatasetToolError(
        f"Unsupported or invalid archive: {archive}. Use ZIP, TAR, TAR.GZ, or TAR.XZ.",
        code="archive_invalid",
    )


def safe_extract_archive(
    archive: Path,
    destination: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    archive_kind = validate_archive_type(archive)
    destination = destination.resolve()
    staging = destination.parent / f".{destination.name}.extracting"
    if destination.exists() and not force:
        raise DatasetToolError(
            f"Raw destination already exists: {destination}. Use --force explicitly to replace it.",
            code="destination_exists",
        )
    if dry_run:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        if archive_kind == "zip":
            _extract_zip(archive, staging)
        else:
            _extract_tar(archive, staging)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination
