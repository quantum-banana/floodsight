"""Unified-space confusion matrix, IoU, mIoU, and Dice metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .supervision import mask_unsupported_logits, validate_labels_against_support


class SegmentationMetrics:
    """Streaming CPU confusion matrix with dataset-aware prediction masking."""

    def __init__(self, *, num_labels: int, ignore_index: int) -> None:
        if num_labels < 2:
            raise ValueError("num_labels must be at least two.")
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_labels, num_labels), dtype=torch.int64)

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        class_availability: Tensor,
    ) -> None:
        validate_labels_against_support(
            labels,
            class_availability,
            ignore_index=self.ignore_index,
        )
        predictions = mask_unsupported_logits(logits, class_availability).argmax(dim=1)
        valid = labels != self.ignore_index
        targets = labels[valid].to(device="cpu", dtype=torch.int64)
        predicted = predictions[valid].to(device="cpu", dtype=torch.int64)
        if targets.numel() == 0:
            return
        indices = targets * self.num_labels + predicted
        counts = torch.bincount(indices, minlength=self.num_labels**2)
        self.confusion_matrix += counts.reshape(self.num_labels, self.num_labels)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion_matrix.to(dtype=torch.float64)
        intersection = matrix.diag()
        target_count = matrix.sum(dim=1)
        prediction_count = matrix.sum(dim=0)
        union = target_count + prediction_count - intersection
        dice_denominator = target_count + prediction_count
        iou = torch.where(union > 0, intersection / union, torch.nan)
        dice = torch.where(dice_denominator > 0, 2 * intersection / dice_denominator, torch.nan)
        total = matrix.sum()
        pixel_accuracy = float(intersection.sum() / total) if total > 0 else None
        return {
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_iou": [None if torch.isnan(item) else float(item) for item in iou],
            "mean_iou": float(torch.nanmean(iou)) if bool((~torch.isnan(iou)).any()) else None,
            "per_class_dice": [None if torch.isnan(item) else float(item) for item in dice],
            "mean_dice": (
                float(torch.nanmean(dice)) if bool((~torch.isnan(dice)).any()) else None
            ),
            "pixel_accuracy": pixel_accuracy,
            "evaluated_pixels": int(total),
        }
