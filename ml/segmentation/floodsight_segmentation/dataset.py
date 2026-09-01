"""Manifest-driven PyTorch dataset for FloodNet and official RescueNet."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler

from .errors import ManifestError
from .manifest import ManifestCollection, ManifestSample, resolve_under_root
from .prepared import PreparedSnapshot, read_verified_pair

SampleTransform = Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class SegmentationManifestDataset(Dataset[dict[str, Any]]):
    """Read only paths enumerated by content-addressed Phase-3 manifests."""

    def __init__(
        self,
        collection: ManifestCollection,
        *,
        data_root: Path,
        supported_class_ids: Mapping[str, frozenset[int]],
        source_to_target_ids: Mapping[str, Mapping[int, int]],
        num_labels: int,
        ignore_index: int,
        transform: SampleTransform,
        verify_sample_hashes: bool,
        prepared_snapshot: PreparedSnapshot | None = None,
    ) -> None:
        if not collection.samples:
            raise ManifestError("The selected manifest collection is empty.")
        self.samples = collection.samples
        self.data_root = data_root.expanduser().resolve(strict=True)
        self.supported_class_ids = supported_class_ids
        self.source_to_target_ids = source_to_target_ids
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.transform = transform
        self.verify_sample_hashes = verify_sample_hashes
        self.prepared_snapshot = prepared_snapshot
        for sample in self.samples:
            if sample.source_dataset not in supported_class_ids:
                raise ManifestError(
                    f"No supervision contract for dataset {sample.source_dataset!r}."
                )
            if sample.source_dataset not in source_to_target_ids:
                raise ManifestError(
                    f"No audited source mapping for dataset {sample.source_dataset!r}."
                )
            if prepared_snapshot is not None:
                prepared = prepared_snapshot.sample_for(sample.sample_id)
                prepared_ids = set(prepared.target_ids) - {self.ignore_index}
                if (
                    prepared.source_dataset != sample.source_dataset
                    or prepared.source_split != sample.source_split
                    or prepared.target_split != sample.target_split
                    or prepared.width != sample.width
                    or prepared.height != sample.height
                    or prepared.ignore_index != sample.ignore_index
                    or prepared.valid_supervision_classes
                    != sample.valid_supervision_classes
                    or prepared.class_counts
                    != tuple(sorted(sample.class_counts.items()))
                    or not prepared_ids <= set(range(self.num_labels))
                    or not prepared_ids <= set(supported_class_ids[sample.source_dataset])
                ):
                    raise ManifestError(
                        f"Prepared snapshot metadata mismatch for {sample.sample_id!r}."
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_pair_strict(
        self, sample: ManifestSample
    ) -> tuple[Image.Image, Image.Image]:
        image_path = resolve_under_root(self.data_root, sample.image_path)
        source_mask_path = resolve_under_root(self.data_root, sample.source_annotation_path)
        mask_path = resolve_under_root(self.data_root, sample.target_annotation_path)
        try:
            image_bytes = image_path.read_bytes()
            source_mask_bytes = source_mask_path.read_bytes()
            mask_bytes = mask_path.read_bytes()
        except OSError as exc:
            raise ManifestError(f"Unable to read sample {sample.sample_id!r}.") from exc
        if self.verify_sample_hashes:
            expected_image_hash = sample.target_image_hash
            expected_mask_hash = sample.target_annotation_hash
            if _sha256_bytes(image_bytes) != expected_image_hash:
                raise ManifestError(f"Image hash mismatch for sample {sample.sample_id!r}.")
            if _sha256_bytes(source_mask_bytes) != sample.annotation_hash:
                raise ManifestError(
                    f"Source annotation hash mismatch for sample {sample.sample_id!r}."
                )
            if _sha256_bytes(mask_bytes) != expected_mask_hash:
                raise ManifestError(f"Mask hash mismatch for sample {sample.sample_id!r}.")
        try:
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as opened_image:
                image = opened_image.convert("RGB")
            with Image.open(BytesIO(source_mask_bytes)) as opened_source_mask:
                source_mask = opened_source_mask.copy()
            with Image.open(BytesIO(mask_bytes)) as opened_mask:
                mask = opened_mask.copy()
        except (OSError, UnidentifiedImageError) as exc:
            raise ManifestError(f"Invalid image or mask for sample {sample.sample_id!r}.") from exc
        if (
            image.size != (sample.width, sample.height)
            or source_mask.size != image.size
            or mask.size != image.size
        ):
            raise ManifestError(
                f"Dimension mismatch for {sample.sample_id!r}: manifest="
                f"{sample.width}x{sample.height}, image={image.size}, "
                f"source_mask={source_mask.size}, mask={mask.size}."
            )
        source_array = np.asarray(source_mask, dtype=np.int64)
        target_array = np.asarray(mask, dtype=np.int64)
        if source_array.ndim != 2 or target_array.ndim != 2:
            raise ManifestError(f"Indexed masks must be single-channel for {sample.sample_id!r}.")
        source_mapping = self.source_to_target_ids[sample.source_dataset]
        source_ids = {int(item) for item in np.unique(source_array).tolist()}
        if not source_ids <= set(source_mapping):
            raise ManifestError(
                f"Source mask has unmapped IDs for {sample.sample_id!r}: "
                f"{sorted(source_ids - set(source_mapping))}."
            )
        remapped = np.full(source_array.shape, sample.ignore_index, dtype=np.int64)
        for source_id, target_id in source_mapping.items():
            remapped[source_array == source_id] = target_id
        if not np.array_equal(remapped, target_array):
            raise ManifestError(
                f"Target mask does not match the frozen mapping for {sample.sample_id!r}."
            )
        unique_ids, unique_counts = np.unique(target_array, return_counts=True)
        recomputed_counts = {
            int(class_id): int(count)
            for class_id, count in zip(unique_ids.tolist(), unique_counts.tolist(), strict=True)
            if int(class_id) != sample.ignore_index
        }
        recomputed_ignored = int(np.count_nonzero(target_array == sample.ignore_index))
        if (
            recomputed_counts != dict(sample.class_counts)
            or recomputed_ignored != sample.ignored_count
        ):
            raise ManifestError(
                f"Target mask class/ignore counts do not match manifest for {sample.sample_id!r}."
            )
        return image, mask

    def read_pair(self, index: int) -> tuple[Image.Image, Image.Image]:
        """Decode a pair without applying stochastic transforms.

        A validated prepared snapshot is the only capability that can select
        the fast path.  Strict mode retains every existing per-fetch check.
        """

        sample = self.samples[index]
        if self.prepared_snapshot is None:
            return self._read_pair_strict(sample)
        return read_verified_pair(self.prepared_snapshot.sample_for(sample.sample_id))

    def transform_pair(
        self,
        index: int,
        image: Image.Image,
        mask: Image.Image,
    ) -> dict[str, Any]:
        """Apply dynamic augmentation and construct one training sample."""

        sample = self.samples[index]
        pixel_values, labels = self.transform(image, mask)
        if labels.ndim != 2 or pixel_values.ndim != 3:
            raise ManifestError(f"Transform returned invalid tensors for {sample.sample_id!r}.")
        supported = self.supported_class_ids[sample.source_dataset]
        if self.prepared_snapshot is None:
            valid_ids = set(int(item) for item in torch.unique(labels).tolist()) - {
                self.ignore_index
            }
            unified_ids = set(range(self.num_labels))
            if not valid_ids <= unified_ids:
                invalid_ids = sorted(valid_ids - unified_ids)
                raise ManifestError(
                    f"Sample {sample.sample_id!r} contains out-of-taxonomy IDs "
                    f"{invalid_ids}."
                )
            if not valid_ids <= supported:
                raise ManifestError(
                    f"Sample {sample.sample_id!r} from {sample.source_dataset!r} contains "
                    f"unsupported IDs {sorted(valid_ids - supported)}."
                )
        class_availability = torch.zeros(self.num_labels, dtype=torch.bool)
        class_availability[list(sorted(supported))] = True
        return {
            "pixel_values": pixel_values,
            "labels": labels.to(dtype=torch.long),
            "class_availability": class_availability,
            "sample_id": sample.sample_id,
            "source_dataset": sample.source_dataset,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, mask = self.read_pair(index)
        return self.transform_pair(index, image, mask)


def build_dataset_balanced_sampler(
    dataset: SegmentationManifestDataset,
    *,
    target_mix: Mapping[str, float],
    generator: torch.Generator,
    replacement: bool,
    num_samples_policy: str,
) -> WeightedRandomSampler:
    """Sample datasets at a declared mixture, independent of their raw sizes."""

    counts = Counter(sample.source_dataset for sample in dataset.samples)
    missing = set(target_mix) - set(counts)
    unexpected = set(counts) - set(target_mix)
    if missing or unexpected:
        raise ManifestError(
            f"Dataset mix mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    if not replacement or num_samples_policy != "training_manifest_size":
        raise ManifestError("Unsupported frozen dataset sampler policy.")
    weights = [
        target_mix[sample.source_dataset] / counts[sample.source_dataset]
        for sample in dataset.samples
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=replacement,
        generator=generator,
    )
