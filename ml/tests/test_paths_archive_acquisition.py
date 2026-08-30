from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from floodsight_data.acquisition import (
    import_archive,
    import_archives,
    import_directory,
    manual_acquisition_status,
)
from floodsight_data.common.archive import (
    safe_extract_archive,
    safe_extract_archives,
    validate_archive_type,
)
from floodsight_data.errors import DatasetToolError
from floodsight_data.paths import DataPaths, resolve_data_root
from floodsight_data.registry import get_dataset


def test_data_root_resolution_from_argument_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = resolve_data_root(tmp_path / "explicit")
    monkeypatch.setenv("FLOODSIGHT_DATA_ROOT", str(tmp_path / "environment"))
    environmental = resolve_data_root()

    assert explicit == (tmp_path / "explicit").resolve()
    assert environmental == (tmp_path / "environment").resolve()


def test_missing_data_root_has_a_useful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOODSIGHT_DATA_ROOT", raising=False)

    with pytest.raises(DatasetToolError, match="FLOODSIGHT_DATA_ROOT") as error:
        resolve_data_root()

    assert error.value.code == "data_root_missing"


def test_data_layout_is_created_only_when_requested(tmp_path: Path) -> None:
    paths = DataPaths(root=tmp_path / "external", cache=tmp_path / "cache")
    assert not paths.root.exists()

    planned = paths.ensure_layout(dry_run=True)
    assert planned
    assert not paths.root.exists()

    paths.ensure_layout()
    assert paths.raw.is_dir()
    assert (paths.processed / "segmentation_v1").is_dir()


def test_safe_zip_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/images/a.txt", "safe")

    destination = safe_extract_archive(archive, tmp_path / "raw")

    assert validate_archive_type(archive) == "zip"
    assert (destination / "train" / "images" / "a.txt").read_text() == "safe"


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


def test_import_archive_accepts_provider_filename_with_spaces(data_paths: DataPaths) -> None:
    archive = _write_zip(
        data_paths.root.parent / "FloodNet dataset export (1).zip",
        {"FloodNet/train/images/a.txt": "image"},
    )

    result = import_archive(data_paths, get_dataset("floodnet"), archive)

    assert Path(result["destination"], "FloodNet", "train", "images", "a.txt").is_file()


def test_multi_archive_import_preserves_both_provider_roots(data_paths: DataPaths) -> None:
    train = _write_zip(
        data_paths.root.parent / "provider train export [final].zip",
        {"VisDrone2019-DET-train/images/train.jpg": "train"},
    )
    val = _write_zip(
        data_paths.root.parent / "provider validation export (1).zip",
        {"VisDrone2019-DET-val/images/val.jpg": "val"},
    )

    result = import_archives(data_paths, get_dataset("visdrone_det"), [train, val])
    destination = Path(result["destination"])
    metadata = json.loads(Path(result["metadata"]).read_text())

    assert (destination / "VisDrone2019-DET-train/images/train.jpg").read_text() == "train"
    assert (destination / "VisDrone2019-DET-val/images/val.jpg").read_text() == "val"
    assert metadata["method"] == "user-multi-archive"
    assert [item["original_filename"] for item in metadata["source_archives"]] == [
        train.name,
        val.name,
    ]
    assert all(item["source_sha256"] for item in metadata["source_archives"])


def test_multi_archive_validates_every_source_before_extraction(tmp_path: Path) -> None:
    safe = _write_zip(tmp_path / "safe.zip", {"safe/file.txt": "safe"})
    unsafe = _write_zip(tmp_path / "unsafe.zip", {"../escape.txt": "unsafe"})
    destination = tmp_path / "raw"

    with pytest.raises(DatasetToolError) as error:
        safe_extract_archives([safe, unsafe], destination)

    assert error.value.code == "archive_path_traversal"
    assert not destination.exists()
    assert not (tmp_path / ".raw.extracting").exists()


def test_multi_archive_rejects_conflicting_paths(tmp_path: Path) -> None:
    first = _write_zip(tmp_path / "first.zip", {"shared/file.txt": "first"})
    second = _write_zip(tmp_path / "second.zip", {"shared/file.txt": "second"})

    with pytest.raises(DatasetToolError) as error:
        safe_extract_archives([first, second], tmp_path / "raw")

    assert error.value.code == "archive_path_conflict"
    assert not (tmp_path / "raw").exists()


def test_multi_archive_accepts_byte_identical_duplicate_paths(tmp_path: Path) -> None:
    first = _write_zip(tmp_path / "first.zip", {"shared/file.txt": "identical"})
    second = _write_zip(tmp_path / "second.zip", {"shared/file.txt": "identical"})

    destination = safe_extract_archives([first, second], tmp_path / "raw")

    assert (destination / "shared/file.txt").read_text() == "identical"


def test_multi_archive_restarts_after_interrupted_staging(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "source.zip", {"dataset/file.txt": "complete"})
    staging = tmp_path / ".raw.extracting"
    staging.mkdir()
    (staging / "stale.txt").write_text("partial")

    destination = safe_extract_archives([archive], tmp_path / "raw")

    assert (destination / "dataset/file.txt").read_text() == "complete"
    assert not (destination / "stale.txt").exists()
    assert not staging.exists()


def test_multi_archive_protects_existing_destination(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "source.zip", {"dataset/new.txt": "new"})
    destination = tmp_path / "raw"
    destination.mkdir()
    (destination / "verified.txt").write_text("keep")

    with pytest.raises(DatasetToolError) as error:
        safe_extract_archives([archive], destination)

    assert error.value.code == "destination_exists"
    assert (destination / "verified.txt").read_text() == "keep"


def test_multi_archive_metadata_is_deterministic_for_same_sources(data_paths: DataPaths) -> None:
    train = _write_zip(data_paths.root.parent / "train.zip", {"train/file.txt": "train"})
    val = _write_zip(data_paths.root.parent / "val.zip", {"val/file.txt": "val"})
    record = get_dataset("visdrone_det")

    first = import_archives(data_paths, record, [train, val])
    first_metadata = json.loads(Path(first["metadata"]).read_text())
    second = import_archives(data_paths, record, [train, val], force=True)
    second_metadata = json.loads(Path(second["metadata"]).read_text())

    assert first_metadata["source_archives"] == second_metadata["source_archives"]
    assert first_metadata["source_set_fingerprint"] == second_metadata["source_set_fingerprint"]
    assert first_metadata["source_fingerprint"] == second_metadata["source_fingerprint"]


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/escape.txt"])
def test_zip_path_traversal_is_rejected(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "unsafe")

    with pytest.raises(DatasetToolError) as error:
        safe_extract_archive(archive, tmp_path / "raw")

    assert error.value.code == "archive_path_traversal"
    assert not (tmp_path / "escape.txt").exists()


def test_tar_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../../escape.txt")
        payload = b"unsafe"
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(DatasetToolError) as error:
        safe_extract_archive(archive, tmp_path / "raw")

    assert error.value.code == "archive_path_traversal"


def test_import_archive_does_not_silently_replace_raw_data(data_paths: DataPaths) -> None:
    archive = data_paths.root.parent / "floodnet.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/images/a.txt", "one")
    record = get_dataset("floodnet")
    first = import_archive(data_paths, record, archive)

    with pytest.raises(DatasetToolError) as error:
        import_archive(data_paths, record, archive)

    assert first["status"] == "IMPORTED_NOT_VALIDATED"
    assert error.value.code == "destination_exists"


def test_import_directory_records_acquisition_metadata(
    data_paths: DataPaths, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    (source / "train" / "images").mkdir(parents=True)
    (source / "train" / "images" / "a.txt").write_text("source")

    result = import_directory(data_paths, get_dataset("rescuenet"), source)
    metadata = json.loads(Path(result["metadata"]).read_text())

    assert result["status"] == "IMPORTED_NOT_VALIDATED"
    assert metadata["method"] == "user-directory"
    assert metadata["status"] == "IMPORTED_NOT_VALIDATED"


def test_gated_acquisition_returns_manual_instructions() -> None:
    status = manual_acquisition_status(get_dataset("rescuenet"))

    assert status["status"] == "MANUAL_ACTION_REQUIRED"
    assert status["license_review_state"] == "REVIEW_REQUIRED"
    assert status["supported_follow_up"] == ["import-archive", "import-directory"]
