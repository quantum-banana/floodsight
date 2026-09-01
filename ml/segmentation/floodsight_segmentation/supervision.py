"""Dataset-aware partial supervision for the unified segmentation taxonomy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .errors import ManifestError


def build_class_availability(
    dataset_ids: Sequence[str],
    *,
    supported_class_ids: Mapping[str, frozenset[int]],
    num_labels: int,
    device: torch.device | None = None,
) -> Tensor:
    """Build the per-sample boolean mask used by loss and metrics."""

    availability = torch.zeros((len(dataset_ids), num_labels), dtype=torch.bool, device=device)
    for row, dataset_id in enumerate(dataset_ids):
        try:
            supported = supported_class_ids[dataset_id]
        except KeyError as exc:
            raise ManifestError(f"No partial-supervision contract for {dataset_id!r}.") from exc
        if not supported or any(class_id < 0 or class_id >= num_labels for class_id in supported):
            raise ManifestError(f"Invalid supported class IDs for {dataset_id!r}.")
        availability[row, list(sorted(supported))] = True
    return availability


def validate_labels_against_support(
    labels: Tensor,
    class_availability: Tensor,
    *,
    ignore_index: int,
) -> None:
    """Fail when any real label is unsupported by its source dataset."""

    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B,H,W], found {tuple(labels.shape)}")
    if class_availability.ndim != 2 or class_availability.shape[0] != labels.shape[0]:
        raise ValueError("class_availability must have shape [B,C].")
    valid = labels != ignore_index
    invalid_range = valid & ((labels < 0) | (labels >= class_availability.shape[1]))
    if bool(invalid_range.any()):
        offending = sorted(int(item) for item in labels[invalid_range].unique().tolist())
        raise ValueError(f"Labels outside the unified taxonomy: {offending}")
    safe_labels = labels.clamp(min=0, max=class_availability.shape[1] - 1)
    supported = class_availability.gather(1, safe_labels.flatten(1)).view_as(labels)
    unsupported = valid & ~supported
    if bool(unsupported.any()):
        details: list[str] = []
        for row in range(labels.shape[0]):
            values = labels[row][unsupported[row]].unique().tolist()
            if values:
                details.append(f"sample {row}: {sorted(int(item) for item in values)}")
        detail_text = "; ".join(details)
        raise ValueError(f"Labels are unsupported by their source dataset ({detail_text}).")


def mask_unsupported_logits(logits: Tensor, class_availability: Tensor) -> Tensor:
    """Remove unsupported classes from each sample's softmax/argmax domain."""

    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B,C,H,W], found {tuple(logits.shape)}")
    if class_availability.shape != logits.shape[:2]:
        raise ValueError(
            "class_availability must match logits [B,C], found "
            f"{tuple(class_availability.shape)} versus {tuple(logits.shape[:2])}."
        )
    if not bool(class_availability.any(dim=1).all()):
        raise ValueError("Every sample must support at least one target class.")
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~class_availability[:, :, None, None], floor)


class PartialCrossEntropyLoss(nn.Module):
    """Cross-entropy over only the classes annotated by each source dataset.

    Unsupported classes are excluded from the softmax denominator. This is
    materially different from ordinary unified-taxonomy cross-entropy: it does
    not train FloodNet pixels against RescueNet-only damage/blocked-road classes,
    or RescueNet pixels against FloodNet-only flooded-road/building classes.
    """

    def __init__(
        self,
        *,
        class_weights: Sequence[float],
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        weights = torch.tensor(tuple(class_weights), dtype=torch.float64)
        if weights.ndim != 1 or weights.numel() < 2:
            raise ValueError("class_weights must be a one-dimensional class vector.")
        if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError("class_weights must contain only finite positive values.")
        self.register_buffer("class_weights", weights, persistent=True)
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        class_availability: Tensor,
    ) -> Tensor:
        validate_labels_against_support(
            labels,
            class_availability,
            ignore_index=self.ignore_index,
        )
        masked_logits = mask_unsupported_logits(logits, class_availability)
        if self.class_weights.numel() != logits.shape[1]:
            raise ValueError(
                "class_weights length must exactly match the unified taxonomy size."
            )
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        per_pixel = F.cross_entropy(
            masked_logits,
            labels,
            weight=weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = labels != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0
        safe_labels = labels.clamp(min=0, max=weights.numel() - 1)
        denominator = weights[safe_labels][valid].sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise ValueError("Valid-pixel class-weight denominator is not positive and finite.")
        return per_pixel[valid].sum() / denominator
