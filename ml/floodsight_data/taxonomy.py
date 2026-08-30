from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from floodsight_data.config import TAXONOMY_ROOT, load_yaml
from floodsight_data.errors import DatasetToolError

IGNORE_INDEX = 255


class MappingAction(StrEnum):
    MAP = "MAP"
    MERGE = "MERGE"
    IGNORE = "IGNORE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class TaxonomyClass:
    class_id: int
    name: str
    color: tuple[int, int, int]
    trainable: bool = True


@dataclass(frozen=True, slots=True)
class MappingEntry:
    source_id: int
    source_name: str
    target_id: int | None
    target_name: str | None
    action: MappingAction
    explanation: str
    source_support: str
    review_status: str
    colors: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class MappingTable:
    dataset_id: str
    task_type: str
    mapping_version: str
    taxonomy_version: str
    real_data_review_required: bool
    entries: tuple[MappingEntry, ...]

    @property
    def by_source_id(self) -> dict[int, MappingEntry]:
        return {entry.source_id: entry for entry in self.entries}

    @property
    def by_color(self) -> dict[tuple[int, int, int], MappingEntry]:
        return {color: entry for entry in self.entries for color in entry.colors}


def load_taxonomy(name: str) -> tuple[str, tuple[TaxonomyClass, ...]]:
    payload = load_yaml(TAXONOMY_ROOT / name)
    version = str(payload.get("version", ""))
    raw_classes = payload.get("classes")
    if not version or not isinstance(raw_classes, list):
        raise DatasetToolError(f"Invalid taxonomy file: {name}", code="taxonomy_invalid")
    classes: list[TaxonomyClass] = []
    for item in raw_classes:
        if not isinstance(item, dict):
            raise DatasetToolError(f"Invalid class in {name}", code="taxonomy_invalid")
        color = tuple(int(channel) for channel in item.get("color", []))
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise DatasetToolError(f"Invalid class color in {name}", code="taxonomy_invalid")
        classes.append(
            TaxonomyClass(
                class_id=int(item["id"]),
                name=str(item["name"]),
                color=color,
                trainable=bool(item.get("trainable", True)),
            )
        )
    ids = [item.class_id for item in classes]
    names = [item.name for item in classes]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise DatasetToolError(f"Duplicate class in {name}", code="taxonomy_invalid")
    return version, tuple(sorted(classes, key=lambda item: item.class_id))


def load_mapping(dataset_id: str) -> MappingTable:
    filename = f"{dataset_id.replace('_det', '').replace('_', '-')}-mapping-v1.yaml"
    payload = load_yaml(TAXONOMY_ROOT / filename)
    raw_entries = payload.get("mappings")
    if not isinstance(raw_entries, list):
        raise DatasetToolError(f"Invalid mapping file: {filename}", code="mapping_invalid")
    entries: list[MappingEntry] = []
    for item in raw_entries:
        try:
            colors = tuple(
                tuple(int(channel) for channel in color) for color in item.get("colors", [])
            )
            if any(len(color) != 3 for color in colors):
                raise ValueError("colors must be RGB triplets")
            target_id = item.get("target_id")
            target_name = item.get("target_name")
            entries.append(
                MappingEntry(
                    source_id=int(item["source_id"]),
                    source_name=str(item["source_name"]),
                    target_id=None if target_id is None else int(target_id),
                    target_name=None if target_name is None else str(target_name),
                    action=MappingAction(item["action"]),
                    explanation=str(item["explanation"]),
                    source_support=str(item["source_support"]),
                    review_status=str(item["review_status"]),
                    colors=colors,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetToolError(
                f"Invalid mapping entry in {filename}: {exc}",
                code="mapping_invalid",
            ) from exc
    source_ids = [entry.source_id for entry in entries]
    colors = [color for entry in entries for color in entry.colors]
    if len(source_ids) != len(set(source_ids)) or len(colors) != len(set(colors)):
        raise DatasetToolError(f"Duplicate source mapping in {filename}", code="mapping_invalid")
    for entry in entries:
        if entry.action in {MappingAction.MAP, MappingAction.MERGE} and (
            entry.target_id is None or entry.target_name is None
        ):
            raise DatasetToolError(
                f"Mapped entry {entry.source_name} lacks a target in {filename}",
                code="mapping_invalid",
            )
    return MappingTable(
        dataset_id=str(payload["dataset_id"]),
        task_type=str(payload["task_type"]),
        mapping_version=str(payload["mapping_version"]),
        taxonomy_version=str(payload["taxonomy_version"]),
        real_data_review_required=bool(payload.get("real_data_review_required", True)),
        entries=tuple(entries),
    )


def validate_mapping_targets(mapping: MappingTable, taxonomy_file: str) -> None:
    _, classes = load_taxonomy(taxonomy_file)
    targets = {item.class_id: item.name for item in classes}
    for entry in mapping.entries:
        if (
            entry.action in {MappingAction.MAP, MappingAction.MERGE}
            and targets.get(entry.target_id) != entry.target_name
        ):
            raise DatasetToolError(
                f"Mapping target mismatch for {entry.source_name}: "
                f"{entry.target_id}/{entry.target_name}",
                code="mapping_invalid",
            )


def mapping_markdown(mapping: MappingTable) -> str:
    rows = [
        f"# {mapping.dataset_id} mapping {mapping.mapping_version}",
        "",
        "| Source ID | Source class | Action | Target ID | Target class | Review | Explanation |",
        "| ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for entry in mapping.entries:
        rows.append(
            "| "
            + " | ".join(
                (
                    str(entry.source_id),
                    entry.source_name,
                    entry.action.value,
                    "" if entry.target_id is None else str(entry.target_id),
                    entry.target_name or "",
                    entry.review_status,
                    entry.explanation.replace("|", "\\|"),
                )
            )
            + " |"
        )
    rows.extend(
        (
            "",
            "> Source evidence is recorded per row. Human mapping, license, and visual review "
            "remain fingerprint-bound requirements before data-verified status.",
            "",
        )
    )
    return "\n".join(rows)


def taxonomy_payload(name: str) -> dict[str, Any]:
    return load_yaml(TAXONOMY_ROOT / name)
