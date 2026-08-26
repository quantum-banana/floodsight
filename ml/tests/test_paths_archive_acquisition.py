from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from floodsight_data.acquisition import import_archive, import_directory, manual_acquisition_status
from floodsight_data.common.archive import safe_extract_archive, validate_archive_type
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
