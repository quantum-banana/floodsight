from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from floodsight_data.errors import BlockingValidationError


@dataclass(frozen=True, slots=True)
class VisDroneObject:
    left: float
    top: float
    width: float
    height: float
    score: int
    class_id: int
    truncation: int
    occlusion: int
    line_number: int


def parse_annotation(path: Path) -> tuple[VisDroneObject, ...]:
    objects: list[VisDroneObject] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BlockingValidationError(
            f"Unable to read VisDrone annotation: {path}",
            code="annotation_corrupt",
        ) from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != 8:
            raise BlockingValidationError(
                f"VisDrone annotation must contain 8 fields at {path}:{line_number}.",
                code="annotation_fields_invalid",
                details=[{"file": str(path), "line": line_number, "field_count": len(fields)}],
            )
        try:
            left, top, width, height = (float(value) for value in fields[:4])
            score, class_id, truncation, occlusion = (int(value) for value in fields[4:])
        except ValueError as exc:
            raise BlockingValidationError(
                f"Non-numeric VisDrone annotation at {path}:{line_number}.",
                code="annotation_fields_invalid",
            ) from exc
        if score not in {0, 1} or truncation not in {0, 1, 2} or occlusion not in {0, 1, 2}:
            raise BlockingValidationError(
                f"Invalid VisDrone metadata at {path}:{line_number}.",
                code="annotation_metadata_invalid",
            )
        objects.append(
            VisDroneObject(
                left,
                top,
                width,
                height,
                score,
                class_id,
                truncation,
                occlusion,
                line_number,
            )
        )
    return tuple(objects)
