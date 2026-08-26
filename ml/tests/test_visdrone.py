from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodsight_data.common.materialize import MaterializationStrategy
from floodsight_data.errors import BlockingValidationError
from floodsight_data.hashing import IntegrityMode
from floodsight_data.paths import DataPaths
from floodsight_data.visdrone.converter import convert_visdrone_dataset
from floodsight_data.visdrone.parser import parse_annotation


def _source(
    paths: DataPaths,
    write_rgb_image: object,
    annotation: str,
    *,
    split: str = "train",
    stem: str = "000001",
) -> tuple[Path, Path]:
    root = paths.raw / "visdrone_det" / f"VisDrone2019-DET-{split}"
    image = root / "images" / f"{stem}.jpg"
    label = root / "annotations" / f"{stem}.txt"
    write_rgb_image(image, (100, 50), 60)
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(annotation, encoding="utf-8")
    return image, label


def test_visdrone_annotation_parser_validates_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "valid.txt"
    path.write_text("10,5,20,10,1,4,1,2\n")

    item = parse_annotation(path)[0]

    assert (item.left, item.top, item.width, item.height) == (10, 5, 20, 10)
    assert (item.score, item.class_id, item.truncation, item.occlusion) == (1, 4, 1, 2)


def test_visdrone_annotation_parser_rejects_bad_field_count(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text("1,2,3\n")

    with pytest.raises(BlockingValidationError) as error:
        parse_annotation(path)

    assert error.value.code == "annotation_fields_invalid"


def test_visdrone_person_merge_vehicle_mapping_and_yolo_normalization(
    data_paths: DataPaths, write_rgb_image: object
) -> None:
    _source(
        data_paths,
        write_rgb_image,
        "10,5,20,10,1,1,0,0\n20,10,10,10,1,2,0,1\n40,10,20,20,1,4,0,0\n60,5,10,10,1,10,1,2\n",
    )

    result = convert_visdrone_dataset(
        data_paths,
        integrity=IntegrityMode.FULL,
        materialization=MaterializationStrategy.COPY,
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    label = data_paths.root / manifest["samples"][0]["target_annotation_path"]
    lines = label.read_text().splitlines()

    assert [line.split()[0] for line in lines] == ["0", "0", "1", "6"]
    assert lines[0] == "0 0.200000 0.200000 0.200000 0.200000"
    assert manifest["samples"][0]["class_counts"] == {"0": 2, "1": 1, "6": 1}
    assert (data_paths.processed / "detection_v1" / "dataset.yaml").is_file()


def test_ignored_regions_and_other_classes_are_not_retained(
    data_paths: DataPaths, write_rgb_image: object
) -> None:
    _source(
        data_paths,
        write_rgb_image,
        "0,0,100,50,1,0,0,0\n10,10,10,10,1,11,0,0\n",
    )

    result = convert_visdrone_dataset(
        data_paths,
        integrity=IntegrityMode.FAST,
        materialization=MaterializationStrategy.MANIFEST_ONLY,
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    sample = manifest["samples"][0]
    label = data_paths.root / sample["target_annotation_path"]

    assert label.read_text() == ""
    assert sample["ignored_count"] == 2
    assert sample["class_counts"] == {}


def test_score_zero_object_is_ignored(data_paths: DataPaths, write_rgb_image: object) -> None:
    _source(data_paths, write_rgb_image, "10,10,10,10,0,1,0,0\n")

    result = convert_visdrone_dataset(
        data_paths,
        integrity=IntegrityMode.FAST,
        materialization=MaterializationStrategy.COPY,
    )
    sample = json.loads(Path(result["manifest"]).read_text())["samples"][0]

    assert sample["ignored_count"] == 1
    assert sample["objects"] == []


@pytest.mark.parametrize(
    "annotation",
    [
        "10,10,0,10,1,4,0,0\n",
        "10,10,-2,10,1,4,0,0\n",
        "200,10,20,10,1,4,0,0\n",
    ],
)
def test_invalid_bounding_boxes_are_blocking(
    data_paths: DataPaths, write_rgb_image: object, annotation: str
) -> None:
    _source(data_paths, write_rgb_image, annotation)

    with pytest.raises(BlockingValidationError) as error:
        convert_visdrone_dataset(
            data_paths,
            integrity=IntegrityMode.FAST,
            materialization=MaterializationStrategy.COPY,
        )

    assert error.value.code == "invalid_bounding_box"


def test_partially_outside_box_is_clamped_and_reported(
    data_paths: DataPaths, write_rgb_image: object
) -> None:
    _source(data_paths, write_rgb_image, "-2,-1,10,10,1,4,1,0\n")

    result = convert_visdrone_dataset(
        data_paths,
        integrity=IntegrityMode.FULL,
        materialization=MaterializationStrategy.COPY,
    )
    sample = json.loads(Path(result["manifest"]).read_text())["samples"][0]
    label = (data_paths.root / sample["target_annotation_path"]).read_text().strip()

    assert sample["objects"][0]["clamped"] is True
    assert label == "1 0.040000 0.090000 0.080000 0.180000"


def test_unsupported_visdrone_class_is_blocking(
    data_paths: DataPaths, write_rgb_image: object
) -> None:
    _source(data_paths, write_rgb_image, "10,10,10,10,1,99,0,0\n")

    with pytest.raises(BlockingValidationError) as error:
        convert_visdrone_dataset(
            data_paths,
            integrity=IntegrityMode.FAST,
            materialization=MaterializationStrategy.COPY,
        )

    assert error.value.code == "unsupported_detection_class"
