from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class TaxonomyClass:
    class_id: int
    name: str
    color: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Taxonomy:
    version: str
    classes: tuple[TaxonomyClass, ...]
    ignore_index: int | None = None

    @property
    def by_id(self) -> dict[int, TaxonomyClass]:
        return {item.class_id: item for item in self.classes}

    @property
    def by_name(self) -> dict[str, TaxonomyClass]:
        return {item.name: item for item in self.classes}


def load_taxonomy(path: Path, *, expected_version: str | None = None) -> Taxonomy:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load taxonomy {path.name}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("classes"), list):
        raise ValueError(f"Taxonomy {path.name} has an invalid structure")
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"Taxonomy {path.name} has no version")
    if expected_version is not None and version != expected_version:
        raise ValueError(f"Expected taxonomy {expected_version}, found {version}")
    classes: list[TaxonomyClass] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in raw["classes"]:
        if not isinstance(item, dict):
            raise ValueError(f"Taxonomy {path.name} contains a non-object class")
        class_id = item.get("id")
        name = item.get("name")
        color = item.get("color")
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not isinstance(name, str)
            or not isinstance(color, list)
            or len(color) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in color)
        ):
            raise ValueError(f"Taxonomy {path.name} contains an invalid class")
        if class_id in seen_ids or name in seen_names:
            raise ValueError(f"Taxonomy {path.name} contains duplicate IDs or names")
        seen_ids.add(class_id)
        seen_names.add(name)
        classes.append(TaxonomyClass(class_id, name, tuple(color)))
    if not classes:
        raise ValueError(f"Taxonomy {path.name} has no classes")
    return Taxonomy(
        version=version,
        classes=tuple(sorted(classes, key=lambda item: item.class_id)),
        ignore_index=raw.get("ignore_index"),
    )
