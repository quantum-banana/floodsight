from __future__ import annotations

import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from floodsight_data.errors import DatasetToolError

MAX_ARCHIVE_MEMBERS = 2_000_000
MAX_EXPANDED_BYTES = 200 * 1024 * 1024 * 1024


def _member_path(name: str, destination: Path) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not pure.parts or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
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


def _is_archive_root_directory(name: str, *, is_directory: bool) -> bool:
    """Recognize a harmless provider-created ZIP root marker such as `/`."""
    return is_directory and not name.replace("\\", "/").strip("/")


def _path_conflict(archive: Path, member_name: str, target: Path) -> DatasetToolError:
    return DatasetToolError(
        f"Archive member conflicts with an extracted path: {member_name} from {archive}",
        code="archive_path_conflict",
        details=[{"archive": str(archive), "member": member_name, "target": str(target)}],
    )


def _ensure_directory(target: Path, *, archive: Path, member_name: str) -> None:
    if target.exists() and not target.is_dir():
        raise _path_conflict(archive, member_name, target)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        raise _path_conflict(archive, member_name, target) from exc


def _files_equal(left: Path, right: Path, *, chunk_size: int = 1024 * 1024) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while left_chunk := left_stream.read(chunk_size):
            if left_chunk != right_stream.read(len(left_chunk)):
                return False
        return not right_stream.read(1)


def _merge_file(
    source: BinaryIO,
    target: Path,
    *,
    archive: Path,
    member_name: str,
) -> None:
    _ensure_directory(target.parent, archive=archive, member_name=member_name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".incoming", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if target.exists():
            if not target.is_file() or not _files_equal(target, temporary):
                raise _path_conflict(archive, member_name, target)
            return
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_zip(archive: Path, destination: Path) -> tuple[int, int]:
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            expanded_bytes = sum(member.file_size for member in members)
            _validate_limits(len(members), expanded_bytes)
            for member in members:
                if _is_archive_root_directory(member.filename, is_directory=member.is_dir()):
                    continue
                mode = member.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise DatasetToolError(
                        f"Archive links and special entries are not allowed: {member.filename}",
                        code="archive_unsafe_member",
                    )
                _member_path(member.filename, destination)
            return len(members), expanded_bytes
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetToolError(
            f"Unable to inspect ZIP archive {archive}: {exc}", code="archive_invalid"
        ) from exc


def _validate_tar(archive: Path, destination: Path) -> tuple[int, int]:
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            members = bundle.getmembers()
            expanded_bytes = sum(member.size for member in members if member.isfile())
            _validate_limits(len(members), expanded_bytes)
            for member in members:
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise DatasetToolError(
                        f"Archive links and special entries are not allowed: {member.name}",
                        code="archive_unsafe_member",
                    )
                if not member.isdir() and not member.isfile():
                    raise DatasetToolError(
                        f"Unsupported archive member: {member.name}",
                        code="archive_unsafe_member",
                    )
                _member_path(member.name, destination)
            return len(members), expanded_bytes
    except (OSError, tarfile.TarError) as exc:
        raise DatasetToolError(
            f"Unable to inspect TAR archive {archive}: {exc}", code="archive_invalid"
        ) from exc


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if _is_archive_root_directory(member.filename, is_directory=member.is_dir()):
                continue
            target = _member_path(member.filename, destination)
            if member.is_dir():
                _ensure_directory(target, archive=archive, member_name=member.filename)
                continue
            with bundle.open(member) as source:
                _merge_file(
                    source,
                    target,
                    archive=archive,
                    member_name=member.filename,
                )


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:*") as bundle:
        for member in bundle.getmembers():
            target = _member_path(member.name, destination)
            if member.isdir():
                _ensure_directory(target, archive=archive, member_name=member.name)
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise DatasetToolError(
                    f"Unable to read archive member: {member.name}",
                    code="archive_invalid",
                )
            with source:
                _merge_file(source, target, archive=archive, member_name=member.name)


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
    return safe_extract_archives(
        [archive],
        destination,
        force=force,
        dry_run=dry_run,
    )


def safe_extract_archives(
    archives: Sequence[Path],
    destination: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """Safely merge archives in staging and atomically publish the finished tree."""
    resolved_archives = [archive.resolve() for archive in archives]
    if not resolved_archives:
        raise DatasetToolError("At least one source archive is required.", code="archive_missing")
    if len(set(resolved_archives)) != len(resolved_archives):
        raise DatasetToolError(
            "The same source archive was provided more than once.",
            code="archive_duplicate_source",
        )
    destination = destination.resolve()
    staging = destination.parent / f".{destination.name}.extracting"
    for archive in resolved_archives:
        if archive == destination or destination in archive.parents:
            raise DatasetToolError(
                f"Source archive overlaps the raw destination: {archive}", code="unsafe_path"
            )
    if destination.exists() and not force:
        raise DatasetToolError(
            f"Raw destination already exists: {destination}. Use --force explicitly to replace it.",
            code="destination_exists",
        )
    archive_kinds = [validate_archive_type(archive) for archive in resolved_archives]
    totals = [
        _validate_zip(archive, staging)
        if archive_kind == "zip"
        else _validate_tar(archive, staging)
        for archive, archive_kind in zip(resolved_archives, archive_kinds, strict=True)
    ]
    _validate_limits(
        sum(member_count for member_count, _ in totals),
        sum(expanded_bytes for _, expanded_bytes in totals),
    )
    if dry_run:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for archive, archive_kind in zip(resolved_archives, archive_kinds, strict=True):
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
