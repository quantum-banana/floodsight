"""Focused synthetic contracts for the launch-verified segmentation fast path."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from floodsight_segmentation import dataset as dataset_module
from floodsight_segmentation import prepared as prepared_module
from floodsight_segmentation.config import (
    AUDITED_SOURCE_TO_TARGET_IDS,
    TARGET_DATASET_SUPPORT,
)
from floodsight_segmentation.dataset import SegmentationManifestDataset
from floodsight_segmentation.errors import ManifestError
from floodsight_segmentation.manifest import (
    PREPARATION_VERSION,
    FrozenManifest,
    ManifestCollection,
    ManifestSample,
)
from floodsight_segmentation.prepared import (
    PreparedCacheBuild,
    PreparedCacheRecord,
    PreparedSnapshot,
    build_prepared_cache_record,
    load_prepared_cache_record,
    verify_prepared_cache_at_launch,
)
from floodsight_segmentation.threaded_loader import ThreadedSampleLoader
from floodsight_segmentation.transforms import PairedSegmentationTransform
from PIL import Image

_IMPLEMENTATION_SHA256 = "d" * 64
_INSTRUCTION_SHA256 = "e" * 64
_MAPPING_SHA256 = (
    "fdfbbba84c1cf8ea0176429b8d236693030abc16452f507c94922cc2f0769760"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class _SyntheticCache:
    data_root: Path
    cache_root: Path
    collection: ManifestCollection
    ledger_path: Path
    stage09_report_path: Path
    artifact_index_path: Path
    build: PreparedCacheBuild
    record: PreparedCacheRecord

    def verify(self) -> PreparedSnapshot:
        return verify_prepared_cache_at_launch(
            self.record,
            collection=self.collection,
            data_root=self.data_root,
            max_workers=2,
        )

    def build_again(self) -> PreparedCacheBuild:
        return build_prepared_cache_record(
            collection=self.collection,
            data_root=self.data_root,
            cache_root=self.cache_root,
            cache_id=self.cache_root.name,
            created_at="2026-09-01T00:00:00Z",
            processed_mask_ledger_path=self.ledger_path,
            processed_mask_ledger_sha256=_sha256(self.ledger_path),
            stage09_report_path=self.stage09_report_path,
            stage09_report_sha256=_sha256(self.stage09_report_path),
            artifact_index_path=self.artifact_index_path,
            artifact_index_sha256=_sha256(self.artifact_index_path),
            prepared_cache_implementation_sha256=_IMPLEMENTATION_SHA256,
            user_instruction_sha256=_INSTRUCTION_SHA256,
        )


def _make_synthetic_cache(tmp_path: Path, *, sample_count: int = 3) -> _SyntheticCache:
    data_root = tmp_path / "dataset"
    raw_images = data_root / "raw/floodnet/images"
    raw_masks = data_root / "raw/floodnet/masks"
    processed_root = data_root / "processed" / PREPARATION_VERSION
    processed_masks = processed_root / "masks/floodnet/train"
    report_root = data_root / "reports/stage09"
    for directory in (raw_images, raw_masks, processed_masks, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    samples: list[ManifestSample] = []
    ledger_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for index in range(sample_count):
        sample_id = f"sample-{index:02d}"
        image_path = raw_images / f"{sample_id}.png"
        source_mask_path = raw_masks / f"{sample_id}.png"
        target_path = processed_masks / f"{sample_id}.png"

        image_array = np.empty((8, 10, 3), dtype=np.uint8)
        image_array[..., 0] = np.arange(10, dtype=np.uint8)[None, :] * 20
        image_array[..., 1] = np.arange(8, dtype=np.uint8)[:, None] * 25
        image_array[..., 2] = index * 30
        source_array = np.full((8, 10), 3, dtype=np.uint8)
        target_array = np.full((8, 10), 3, dtype=np.uint8)
        Image.fromarray(image_array, mode="RGB").save(image_path)
        Image.fromarray(source_array, mode="L").save(source_mask_path)
        Image.fromarray(target_array, mode="L").save(target_path)

        image_relative = image_path.relative_to(data_root).as_posix()
        source_relative = source_mask_path.relative_to(data_root).as_posix()
        target_relative = target_path.relative_to(data_root).as_posix()
        target_sha256 = _sha256(target_path)
        samples.append(
            ManifestSample(
                sample_id=sample_id,
                source_dataset="floodnet",
                source_split="train",
                target_split="train",
                image_path=image_relative,
                source_annotation_path=source_relative,
                target_annotation_path=target_relative,
                width=10,
                height=8,
                image_hash=_sha256(image_path),
                annotation_hash=_sha256(source_mask_path),
                target_image_hash=_sha256(image_path),
                target_annotation_hash=target_sha256,
                class_counts={3: 80},
                ignored_count=0,
                invalid_count=0,
                preparation_version=PREPARATION_VERSION,
                taxonomy_version="segmentation-taxonomy-v2",
                source_schema="floodnet-supervised-v1.0-indexed-mask-ids-0-9",
                target_mapping_version="floodnet-mapping-v2",
                target_mapping_sha256=_MAPPING_SHA256,
                valid_supervision_classes=tuple(sorted(TARGET_DATASET_SUPPORT["floodnet"])),
                ignore_index=255,
                ignore_semantics="fixture-ignore-semantics",
                exclusion_status="INCLUDED",
                exclusion_reason="",
            )
        )
        ledger_rows.append(
            {
                "bytes": target_path.stat().st_size,
                "dtype": "uint8",
                "height": 8,
                "ignored_count": 0,
                "invalid_count": 0,
                "mode": "L",
                "path": target_relative,
                "sample_id": sample_id,
                "sha256": target_sha256,
                "source_dataset": "floodnet",
                "target_ids": [3],
                "target_split": "train",
                "width": 10,
            }
        )
        artifacts.append(
            {
                "category": "processed_mask",
                "path": str(target_path),
                "bytes": target_path.stat().st_size,
                "sha256": target_sha256,
            }
        )

    manifest_path = report_root / "floodnet_train_manifest.json"
    _write_json(manifest_path, {"fixture": "prepared-fast-path"})
    manifest_sha256 = _sha256(manifest_path)
    manifest_fingerprint = "b" * 64
    frozen_manifest = FrozenManifest(
        path=manifest_path.resolve(),
        sha256=manifest_sha256,
        manifest_id="prepared-fast-path-fixture",
        dataset_id="floodnet",
        taxonomy_version="segmentation-taxonomy-v2",
        integrity_mode="full",
        fingerprint=manifest_fingerprint,
        samples=tuple(samples),
    )
    collection = ManifestCollection(
        manifests=(frozen_manifest,),
        samples=tuple(samples),
    )

    ledger_path = report_root / "processed-mask-manifest.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    stage09_report_path = report_root / "stage09-report.json"
    _write_json(
        stage09_report_path,
        {
            "preparation_version": PREPARATION_VERSION,
            "processed_mask_count": sample_count,
            "processed_mask_tree_fingerprint": "c" * 64,
            "processed_root": str(processed_root.resolve()),
            "checks": {
                "every_generated_mask_hash_id_and_dimension_validated": True,
                "every_manifest_exact_external_schema_validated": True,
                "every_manifest_fingerprint_recomputed_with_locked_helper": True,
                "every_manifest_loaded_with_full_integrity_contract": True,
                "every_source_hash_validated": True,
                "every_source_mask_id_and_dimension_validated": True,
            },
            "manifests": [
                {
                    "path": str(manifest_path.resolve()),
                    "sha256": manifest_sha256,
                    "fingerprint": manifest_fingerprint,
                    "dataset_id": "floodnet",
                    "sample_count": sample_count,
                }
            ],
        },
    )
    artifact_index_path = report_root / "artifact-index.json"
    _write_json(
        artifact_index_path,
        {"artifact_count": len(artifacts), "artifacts": artifacts},
    )

    cache_parent = tmp_path / "prepared-caches"
    cache_parent.mkdir()
    cache_root = cache_parent / "fastpath-fixture-v1"
    build = build_prepared_cache_record(
        collection=collection,
        data_root=data_root,
        cache_root=cache_root,
        cache_id=cache_root.name,
        created_at="2026-09-01T00:00:00Z",
        processed_mask_ledger_path=ledger_path,
        processed_mask_ledger_sha256=_sha256(ledger_path),
        stage09_report_path=stage09_report_path,
        stage09_report_sha256=_sha256(stage09_report_path),
        artifact_index_path=artifact_index_path,
        artifact_index_sha256=_sha256(artifact_index_path),
        prepared_cache_implementation_sha256=_IMPLEMENTATION_SHA256,
        user_instruction_sha256=_INSTRUCTION_SHA256,
    )
    record = load_prepared_cache_record(
        build.record_path,
        expected_sha256=build.record_sha256,
        collection=collection,
        data_root=data_root,
    )
    return _SyntheticCache(
        data_root=data_root,
        cache_root=cache_root,
        collection=collection,
        ledger_path=ledger_path,
        stage09_report_path=stage09_report_path,
        artifact_index_path=artifact_index_path,
        build=build,
        record=record,
    )


def _dataset(
    fixture: _SyntheticCache,
    *,
    snapshot: PreparedSnapshot | None,
    training: bool,
) -> SegmentationManifestDataset:
    return SegmentationManifestDataset(
        fixture.collection,
        data_root=fixture.data_root,
        supported_class_ids=TARGET_DATASET_SUPPORT,
        source_to_target_ids=AUDITED_SOURCE_TO_TARGET_IDS,
        num_labels=16,
        ignore_index=255,
        transform=PairedSegmentationTransform(
            height=7,
            width=9,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            training=training,
            scale=(0.55, 1.0),
            ratio=(0.75, 1.25),
            horizontal_flip_probability=0.5,
        ),
        verify_sample_hashes=True,
        prepared_snapshot=snapshot,
    )


def test_build_load_and_launch_verify_synthetic_cache(tmp_path: Path) -> None:
    fixture = _make_synthetic_cache(tmp_path)
    snapshot = fixture.verify()

    assert fixture.build.sample_count == len(fixture.collection.samples) == 3
    assert fixture.build.record_path.is_file()
    assert fixture.record.sha256 == fixture.build.record_sha256
    assert snapshot.record_sha256 == fixture.build.record_sha256
    assert snapshot.manifest_set_fingerprint == fixture.collection.set_fingerprint
    assert snapshot.prepared_cache_implementation_sha256 == _IMPLEMENTATION_SHA256
    assert len(snapshot.samples) == 3
    for sample in snapshot.samples:
        assert sample.image.sha256 == fixture.collection.samples[
            int(sample.sample_id.rsplit("-", 1)[1])
        ].target_image_hash
        assert sample.cache_target.sha256 == sample.stage09_target_sha256
        assert sample.cache_target.path.is_relative_to(fixture.cache_root)


def test_strict_and_prepared_paths_have_exact_pixels_targets_and_rng_tensors(
    tmp_path: Path,
) -> None:
    fixture = _make_synthetic_cache(tmp_path, sample_count=1)
    snapshot = fixture.verify()
    strict = _dataset(fixture, snapshot=None, training=True)
    prepared = _dataset(fixture, snapshot=snapshot, training=True)

    strict_image, strict_target = strict.read_pair(0)
    prepared_image, prepared_target = prepared.read_pair(0)
    assert np.array_equal(np.asarray(strict_image), np.asarray(prepared_image))
    assert np.array_equal(np.asarray(strict_target), np.asarray(prepared_target))

    torch.manual_seed(8731)
    strict_item = strict[0]
    strict_rng_tail = torch.rand(4)
    torch.manual_seed(8731)
    prepared_item = prepared[0]
    prepared_rng_tail = torch.rand(4)

    assert torch.equal(strict_item["pixel_values"], prepared_item["pixel_values"])
    assert torch.equal(strict_item["labels"], prepared_item["labels"])
    assert torch.equal(
        strict_item["class_availability"], prepared_item["class_availability"]
    )
    assert strict_item["sample_id"] == prepared_item["sample_id"]
    assert strict_item["source_dataset"] == prepared_item["source_dataset"]
    assert torch.equal(strict_rng_tail, prepared_rng_tail)


@pytest.mark.parametrize("corruption", ["image", "source-mask", "cache-target"])
def test_launch_verification_rejects_prelaunch_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    fixture = _make_synthetic_cache(tmp_path, sample_count=1)
    record_sample = fixture.record.samples[0]
    corrupt_path = {
        "image": record_sample.image_path,
        "source-mask": record_sample.source_annotation_path,
        "cache-target": record_sample.cache_target_path,
    }[corruption]
    corrupt_path.write_bytes(corrupt_path.read_bytes() + b"corruption")

    with pytest.raises(ManifestError, match="mismatch"):
        fixture.verify()


@pytest.mark.parametrize("target", ["image", "cache-target"])
def test_prepared_fetch_rejects_postlaunch_stat_drift(
    tmp_path: Path,
    target: str,
) -> None:
    fixture = _make_synthetic_cache(tmp_path, sample_count=1)
    snapshot = fixture.verify()
    sample = snapshot.samples[0]
    drift_path = sample.image.path if target == "image" else sample.cache_target.path
    before = drift_path.stat()
    os.utime(
        drift_path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(ManifestError, match="changed after launch verification"):
        _dataset(fixture, snapshot=snapshot, training=False)[0]


def test_prepared_fetch_skips_source_mask_hash_remap_counts_and_unique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_synthetic_cache(tmp_path, sample_count=1)
    snapshot = fixture.verify()
    prepared = _dataset(fixture, snapshot=snapshot, training=False)
    source_mask = fixture.record.samples[0].source_annotation_path
    source_mask.unlink()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("strict-only work reached the prepared per-fetch hot path")

    class _NoRemapItems(dict[int, int]):
        def items(self) -> Any:
            return forbidden()

    prepared.source_to_target_ids = {
        "floodnet": _NoRemapItems(AUDITED_SOURCE_TO_TARGET_IDS["floodnet"])
    }
    opened: list[Path] = []
    original_open = prepared_module._open_readonly

    def tracking_open(path: Path, *, label: str) -> tuple[int, os.stat_result]:
        opened.append(path)
        return original_open(path, label=label)

    monkeypatch.setattr(prepared_module, "_open_readonly", tracking_open)
    monkeypatch.setattr(prepared_module, "_hash_open_file", forbidden)
    monkeypatch.setattr(dataset_module, "_sha256_bytes", forbidden)
    monkeypatch.setattr(dataset_module.np, "unique", forbidden)
    monkeypatch.setattr(dataset_module.np, "array_equal", forbidden)
    monkeypatch.setattr(dataset_module.np, "full", forbidden)
    monkeypatch.setattr(dataset_module.torch, "unique", forbidden)

    item = prepared[0]

    assert item["sample_id"] == "sample-00"
    assert opened == [snapshot.samples[0].image.path, snapshot.samples[0].cache_target.path]
    assert source_mask not in opened


class _FixedSampler(Sequence[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self._indices = tuple(indices)

    def __getitem__(self, index: int) -> int:
        return self._indices[index]

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)


class _ThreadProbeDataset:
    def __init__(self) -> None:
        self.read_threads: list[int] = []
        self.transform_threads: list[int] = []

    def read_pair(self, index: int) -> tuple[int, int]:
        self.read_threads.append(threading.get_ident())
        time.sleep(0.002 * (index % 3))
        return index, -index

    def transform_pair(self, index: int, image: int, mask: int) -> dict[str, Any]:
        self.transform_threads.append(threading.get_ident())
        assert (image, mask) == (index, -index)
        return {"index": index, "rng": torch.rand(())}


def _run_threaded_probe(seed: int) -> tuple[list[int], torch.Tensor, _ThreadProbeDataset]:
    probe = _ThreadProbeDataset()
    loader = ThreadedSampleLoader(
        probe,
        sampler=_FixedSampler((4, 1, 3, 0, 2)),
        batch_size=2,
        num_threads=3,
        prefetch_samples=4,
        pin_memory=False,
    )
    assert len(loader) == 3
    torch.manual_seed(seed)
    batches = list(loader)
    indices = [int(value) for batch in batches for value in batch["index"]]
    random_values = torch.cat([batch["rng"] for batch in batches])
    return indices, random_values, probe


def test_threaded_loader_preserves_order_main_thread_rng_and_cleans_up() -> None:
    main_thread = threading.get_ident()
    first_indices, first_rng, first_probe = _run_threaded_probe(912)
    second_indices, second_rng, second_probe = _run_threaded_probe(912)

    assert first_indices == second_indices == [4, 1, 3, 0, 2]
    assert torch.equal(first_rng, second_rng)
    assert first_probe.read_threads
    assert all(thread_id != main_thread for thread_id in first_probe.read_threads)
    assert first_probe.transform_threads == [main_thread] * 5
    assert second_probe.transform_threads == [main_thread] * 5

    early_probe = _ThreadProbeDataset()
    early_loader = ThreadedSampleLoader(
        early_probe,
        sampler=_FixedSampler(tuple(range(8))),
        batch_size=2,
        num_threads=2,
        prefetch_samples=4,
        drop_last=True,
    )
    assert len(early_loader) == 4
    iterator = iter(early_loader)
    with iterator:
        assert next(iterator)["index"].tolist() == [0, 1]
    assert iterator._closed is True
    assert not any(
        thread.name.startswith("floodsight-segmentation-reader")
        for thread in threading.enumerate()
    )


def test_prepared_cache_build_refuses_record_or_target_overwrite(tmp_path: Path) -> None:
    fixture = _make_synthetic_cache(tmp_path, sample_count=1)
    record_before = fixture.build.record_path.read_bytes()
    target_before = fixture.record.samples[0].cache_target_path.read_bytes()

    with pytest.raises(ManifestError, match="already exists; refusing overwrite"):
        fixture.build_again()

    assert fixture.build.record_path.read_bytes() == record_before
    assert fixture.record.samples[0].cache_target_path.read_bytes() == target_before
