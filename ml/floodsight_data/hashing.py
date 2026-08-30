from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any


class IntegrityMode(StrEnum):
    FAST = "fast"
    FULL = "full"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_sample_id(dataset_id: str, split: str, relative_image: str) -> str:
    suffix = stable_digest([dataset_id, split, relative_image])[:20]
    return f"{dataset_id}-{split}-{suffix}"


def file_integrity(
    path: Path,
    *,
    relative_path: str,
    mode: IntegrityMode,
    annotation: bool = False,
    precomputed_sha256: str | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": relative_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if mode is IntegrityMode.FULL or annotation:
        payload["sha256"] = precomputed_sha256 or sha256_file(path)
    return payload


def dataset_fingerprint(
    source_records: Iterable[dict[str, Any]],
    *,
    taxonomy_version: str,
    mapping_hashes: dict[str, str],
    preparation: dict[str, Any],
    tool_version: str,
    git_commit: str,
) -> str:
    return stable_digest(
        {
            "source_records": sorted(source_records, key=lambda item: str(item.get("path", ""))),
            "taxonomy_version": taxonomy_version,
            "mapping_hashes": dict(sorted(mapping_hashes.items())),
            "preparation": preparation,
            "tool_version": tool_version,
            "git_commit": git_commit,
        }
    )
