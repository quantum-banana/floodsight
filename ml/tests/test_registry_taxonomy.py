from floodsight_data.registry import LicenseReviewState, TaskType, get_dataset, load_registry
from floodsight_data.taxonomy import (
    load_mapping,
    load_taxonomy,
    mapping_markdown,
    taxonomy_payload,
    validate_mapping_targets,
)


def test_registry_parses_all_three_typed_datasets() -> None:
    registry = load_registry()

    assert set(registry) == {"floodnet", "rescuenet", "visdrone_det"}
    assert registry["floodnet"].task_type is TaskType.SEMANTIC_SEGMENTATION
    assert registry["visdrone_det"].task_type is TaskType.AERIAL_DETECTION


def test_registry_preserves_license_review_caution() -> None:
    rescuenet = get_dataset("rescuenet")

    assert rescuenet.license_review_state is LicenseReviewState.REVIEW_REQUIRED
    assert rescuenet.commercial_use_requires_review is True
    assert "creativecommons.org/licenses/by-nc-nd" in rescuenet.license_reference


def test_taxonomy_ids_and_ignore_index_are_stable() -> None:
    payload = taxonomy_payload("segmentation-taxonomy-v1.yaml")
    version, classes = load_taxonomy("segmentation-taxonomy-v1.yaml")

    assert version == "segmentation-taxonomy-v1"
    assert payload["ignore_index"] == 255
    assert [(item.class_id, item.name) for item in classes] == [
        (0, "background_other"),
        (1, "water"),
        (2, "road_clear"),
        (3, "road_flooded"),
        (4, "road_blocked"),
        (5, "building_normal"),
        (6, "building_flooded"),
        (7, "building_minor_damage"),
        (8, "building_major_damage"),
        (9, "building_destroyed"),
        (10, "vehicle"),
        (11, "vegetation"),
    ]


def test_product_taxonomy_does_not_make_debris_trainable() -> None:
    product = taxonomy_payload("product-taxonomy-v1.yaml")
    segmentation = taxonomy_payload("segmentation-taxonomy-v1.yaml")
    debris = next(item for item in product["classes"] if item["name"] == "debris_landslide")

    assert debris["trainable"] is False
    assert "debris_landslide" not in {item["name"] for item in segmentation["classes"]}


def test_flooded_and_blocked_roads_remain_distinct() -> None:
    floodnet = load_mapping("floodnet")
    rescuenet = load_mapping("rescuenet")

    assert floodnet.by_source_id[3].target_name == "road_flooded"
    assert floodnet.by_source_id[3].target_id == 3
    assert rescuenet.by_source_id[8].target_name == "road_blocked"
    assert rescuenet.by_source_id[8].target_id == 4


def test_detection_mapping_matches_visdrone_source_ids() -> None:
    mapping = load_mapping("visdrone_det")

    assert mapping.by_source_id[0].target_id is None
    assert mapping.by_source_id[1].target_name == "person"
    assert mapping.by_source_id[2].target_name == "person"
    assert mapping.by_source_id[4].target_name == "car"
    assert mapping.by_source_id[8].target_name == "tricycle"
    assert mapping.by_source_id[10].target_name == "motorcycle"
    assert mapping.by_source_id[11].target_id is None


def test_all_mapping_targets_validate_and_render_review_tables() -> None:
    for dataset_id, taxonomy in (
        ("floodnet", "segmentation-taxonomy-v1.yaml"),
        ("rescuenet", "segmentation-taxonomy-v1.yaml"),
        ("visdrone_det", "detection-taxonomy-v1.yaml"),
    ):
        mapping = load_mapping(dataset_id)
        validate_mapping_targets(mapping, taxonomy)
        table = mapping_markdown(mapping)
        assert "| Source ID |" in table
        assert "Real source files" in table
