from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from floodsight_data.common.images import convert_source_mask, read_source_mask
from floodsight_data.common.materialize import MaterializationStrategy
from floodsight_data.common.segmentation_converter import convert_segmentation_dataset
from floodsight_data.errors import BlockingValidationError
from floodsight_data.hashing import IntegrityMode, sha256_file
from floodsight_data.manifests import validate_schema
from floodsight_data.paths import DataPaths
from floodsight_data.taxonomy import MappingAction, MappingEntry, MappingTable, load_mapping


def _source(
    paths: DataPaths,
    write_rgb_image: object,
    write_indexed_mask: object,
    *,
    dataset_id: str = "floodnet",
    split: str = "train",
    mask: np.ndarray | None = None,
    image_size: tuple[int, int] = (8, 6),
) -> tuple[Path, Path]:
    image_path = paths.raw / dataset_id / split / "images" / "sample.jpg"
    mask_path = paths.raw / dataset_id / split / "masks" / "sample.png"
    write_rgb_image(image_path, image_size, 80)
    values = mask if mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    write_indexed_mask(mask_path, values)
    return image_path, mask_path


def test_floodnet_indexed_mask_parsing(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    values = np.array([[0, 1, 3], [4, 5, 7]], dtype=np.uint8)
    Image.fromarray(values, mode="L").save(path)

    parsed = read_source_mask(path, load_mapping("floodnet"))

    assert np.array_equal(parsed, values)


def test_rescuenet_rgb_mask_parsing(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    rgb = np.array([[[0, 0, 0], [255, 0, 0]], [[140, 140, 140], [160, 150, 20]]], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)

    parsed = read_source_mask(path, load_mapping("rescuenet"))

    assert parsed.tolist() == [[0, 5], [7, 8]]


def test_palette_mask_uses_known_palette_colors_when_indices_are_not_ids(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    image = Image.fromarray(np.array([[42, 42], [42, 42]], dtype=np.uint8), mode="P")
    palette = [0] * (256 * 3)
    palette[42 * 3 : 42 * 3 + 3] = [255, 0, 0]
    image.putpalette(palette)
    image.save(path)

    parsed = read_source_mask(path, load_mapping("floodnet"))

    assert set(np.unique(parsed).tolist()) == {1}


def test_unknown_mask_id_is_blocking(tmp_path: Path) -> None:
    path = tmp_path / "unknown.png"
    Image.fromarray(np.full((2, 2), 99, dtype=np.uint8), mode="L").save(path)

    with pytest.raises(BlockingValidationError) as error:
        read_source_mask(path, load_mapping("floodnet"))

    assert error.value.code == "unknown_mask_ids"
    assert "99" in str(error.value)


def test_unknown_mask_color_is_blocking(tmp_path: Path) -> None:
    path = tmp_path / "unknown.png"
    Image.new("RGB", (2, 2), (12, 34, 56)).save(path)

    with pytest.raises(BlockingValidationError) as error:
        read_source_mask(path, load_mapping("rescuenet"))

    assert error.value.code == "unknown_mask_colors"


def test_ignore_action_writes_255_and_validates_target_ids() -> None:
    mapping = MappingTable(
        dataset_id="fixture",
        task_type="SEMANTIC_SEGMENTATION",
        mapping_version="fixture-v1",
        taxonomy_version="segmentation-taxonomy-v1",
        real_data_review_required=False,
        entries=(
            MappingEntry(0, "background", 0, "background_other", MappingAction.MAP, "", "", "", ()),
            MappingEntry(1, "unsupported", None, None, MappingAction.IGNORE, "", "", "", ()),
        ),
    )
    source = np.array([[0, 1], [1, 0]], dtype=np.int32)

    target, counts, ignored = convert_source_mask(
        source, mapping, path=Path("fixture.png"), valid_target_ids={0}
    )

    assert target.tolist() == [[0, 255], [255, 0]]
    assert counts == {0: 2}
    assert ignored == 2


def test_segmentation_conversion_rejects_dimension_mismatch(
    data_paths: DataPaths, write_rgb_image: object, write_indexed_mask: object
) -> None:
    _source(
        data_paths,
        write_rgb_image,
        write_indexed_mask,
        mask=np.zeros((4, 4), dtype=np.uint8),
        image_size=(8, 6),
    )

    with pytest.raises(BlockingValidationError) as error:
        convert_segmentation_dataset(
            data_paths,
            "floodnet",
            integrity=IntegrityMode.FAST,
            materialization=MaterializationStrategy.COPY,
        )

    assert error.value.code == "dimension_mismatch"


def test_segmentation_conversion_is_atomic_preserves_raw_and_writes_manifest(
    data_paths: DataPaths, write_rgb_image: object, write_indexed_mask: object
) -> None:
    values = np.array(
        [
            [0, 0, 1, 1, 3, 3, 5, 5],
            [0, 0, 1, 1, 3, 3, 5, 5],
            [2, 2, 4, 4, 6, 6, 7, 7],
            [2, 2, 4, 4, 6, 6, 7, 7],
            [8, 8, 9, 9, 0, 0, 0, 0],
            [8, 8, 9, 9, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    image, source_mask = _source(data_paths, write_rgb_image, write_indexed_mask, mask=values)
    image_before = sha256_file(image)
    mask_before = sha256_file(source_mask)

    result = convert_segmentation_dataset(
        data_paths,
        "floodnet",
        integrity=IntegrityMode.FULL,
        materialization=MaterializationStrategy.COPY,
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    output_mask = data_paths.root / manifest["samples"][0]["target_annotation_path"]

    assert result["sample_count"] == 1
    assert Image.open(output_mask).mode == "L"
    assert set(np.unique(np.asarray(Image.open(output_mask))).tolist()) <= set(range(12)) | {255}
    assert sha256_file(image) == image_before
    assert sha256_file(source_mask) == mask_before
    assert not list(output_mask.parent.glob("*.tmp"))
    validate_schema(manifest, "dataset-manifest.schema.json")
    lock = json.loads((data_paths.locks / "floodnet-segmentation_v1.json").read_text())
    validate_schema(lock, "dataset-lock.schema.json")


def test_segmentation_conversion_resumes_identical_outputs(
    data_paths: DataPaths, write_rgb_image: object, write_indexed_mask: object
) -> None:
    _source(data_paths, write_rgb_image, write_indexed_mask)
    options = {
        "integrity": IntegrityMode.FAST,
        "materialization": MaterializationStrategy.HARDLINK,
    }

    first = convert_segmentation_dataset(data_paths, "floodnet", **options)
    first_manifest = Path(first["manifest"]).read_text()
    second = convert_segmentation_dataset(data_paths, "floodnet", **options)

    assert first["fingerprint"] == second["fingerprint"]
    assert first["resumed_count"] == 0
    assert second["resumed_count"] == 1
    assert Path(second["manifest"]).read_text() == first_manifest


def test_missing_image_mask_pair_blocks_conversion(
    data_paths: DataPaths, write_rgb_image: object
) -> None:
    write_rgb_image(data_paths.raw / "floodnet" / "train" / "images" / "orphan.jpg")

    with pytest.raises(BlockingValidationError) as error:
        convert_segmentation_dataset(
            data_paths,
            "floodnet",
            integrity=IntegrityMode.FAST,
            materialization=MaterializationStrategy.MANIFEST_ONLY,
        )

    assert error.value.code == "pairing_failed"
