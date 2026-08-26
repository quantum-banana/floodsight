from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from floodsight_data.common.atomic import atomic_build, atomic_write_json
from floodsight_data.errors import DatasetToolError
from floodsight_data.manifests import read_json
from floodsight_data.paths import DataPaths
from floodsight_data.taxonomy import IGNORE_INDEX, load_taxonomy


def _select(samples: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise DatasetToolError(
            "Inspection count must be positive.", code="inspection_count_invalid"
        )
    warning_samples = sorted(
        (
            sample
            for sample in samples
            if sample.get("invalid_count", 0)
            or any(item.get("clamped") for item in sample.get("objects", []))
        ),
        key=lambda sample: sample["sample_id"],
    )
    ordered = sorted(
        samples,
        key=lambda sample: (
            len([value for value in sample["class_counts"].values() if value]),
            sample["sample_id"],
        ),
        reverse=True,
    )
    rare = [sample for sample in ordered if sample not in warning_samples][
        : min(len(ordered), max(1, count // 3))
    ]
    priority = (warning_samples + rare)[:count]
    remaining = [sample for sample in samples if sample not in priority]
    random.Random(seed).shuffle(remaining)
    return (priority + remaining)[:count]


def _fit(image: Image.Image, size: tuple[int, int] = (480, 320)) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (19, 25, 34))
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _segmentation_panel(paths: DataPaths, sample: dict[str, Any]) -> Image.Image:
    _, classes = load_taxonomy("segmentation-taxonomy-v1.yaml")
    colors = {item.class_id: item.color for item in classes}
    image = Image.open(paths.root / sample["image_path"]).convert("RGB")
    mask_image = Image.open(paths.root / sample["target_annotation_path"])
    mask = np.asarray(mask_image, dtype=np.uint8)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in colors.items():
        color[mask == class_id] = rgb
    color[mask == IGNORE_INDEX] = (255, 0, 255)
    colored = Image.fromarray(color, mode="RGB")
    blend = Image.blend(image, colored, 0.45)
    panels = [_fit(image), _fit(colored), _fit(blend)]
    output = Image.new("RGB", (480 * 3, 365), (12, 17, 24))
    draw = ImageDraw.Draw(output)
    for index, panel in enumerate(panels):
        output.paste(panel, (index * 480, 45))
    draw.text(
        (12, 12),
        f"{sample['sample_id']} | {sample['source_dataset']} | "
        f"{sample['source_split']} | image / mask / overlay",
        fill=(235, 241, 248),
        font=ImageFont.load_default(),
    )
    return output


def _detection_panel(paths: DataPaths, sample: dict[str, Any]) -> Image.Image:
    _, classes = load_taxonomy("detection-taxonomy-v1.yaml")
    names = {item.class_id: item.name for item in classes}
    colors = {item.class_id: item.color for item in classes}
    image = Image.open(paths.root / sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_path = paths.root / sample["target_annotation_path"]
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, cx_text, cy_text, width_text, height_text = raw_line.split()
        class_id = int(class_id_text)
        cx, cy, box_width, box_height = map(float, (cx_text, cy_text, width_text, height_text))
        left = (cx - box_width / 2) * image.width
        top = (cy - box_height / 2) * image.height
        right = (cx + box_width / 2) * image.width
        bottom = (cy + box_height / 2) * image.height
        draw.rectangle((left, top, right, bottom), outline=colors[class_id], width=3)
        draw.text((left + 2, top + 2), names[class_id], fill=colors[class_id])
    panel = _fit(image, (960, 540))
    output = Image.new("RGB", (960, 585), (12, 17, 24))
    output.paste(panel, (0, 45))
    ImageDraw.Draw(output).text(
        (12, 12),
        f"{sample['sample_id']} | {sample['source_dataset']} | "
        f"{sample['source_split']} | objects="
        f"{sum(sample['class_counts'].values())}",
        fill=(235, 241, 248),
        font=ImageFont.load_default(),
    )
    return output


def generate_inspection(
    paths: DataPaths,
    dataset_id: str,
    *,
    split: str | None,
    count: int,
    seed: int = 1337,
) -> dict[str, Any]:
    version = "detection_v1" if dataset_id == "visdrone_det" else "segmentation_v1"
    manifest_path = paths.manifests / f"{dataset_id}-{version}.json"
    if not manifest_path.is_file():
        raise DatasetToolError(
            f"Prepared manifest is missing: {manifest_path}. Run convert or prepare first.",
            code="manifest_missing",
        )
    manifest = read_json(manifest_path)
    samples = [
        sample for sample in manifest["samples"] if split is None or sample["target_split"] == split
    ]
    if not samples:
        raise DatasetToolError(
            "No samples match the requested inspection split.", code="split_empty"
        )
    selected = _select(samples, min(count, len(samples)), seed)
    panels = [
        _detection_panel(paths, sample)
        if dataset_id == "visdrone_det"
        else _segmentation_panel(paths, sample)
        for sample in selected
    ]
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (cell_width, cell_height * len(panels)), (8, 12, 18))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * cell_height))
    output_dir = paths.inspections / dataset_id
    output_path = output_dir / f"{split or 'all'}-contact-sheet.png"

    def build(temporary: Path) -> None:
        sheet.save(temporary, format="PNG", compress_level=6)

    atomic_build(output_path, build)
    index = {
        "dataset_id": dataset_id,
        "split": split,
        "seed": seed,
        "count": len(selected),
        "contact_sheet": str(output_path.relative_to(paths.root).as_posix()),
        "samples": [sample["sample_id"] for sample in selected],
    }
    index_path = output_dir / f"{split or 'all'}-inspection-index.json"
    atomic_write_json(index_path, index)
    return index
