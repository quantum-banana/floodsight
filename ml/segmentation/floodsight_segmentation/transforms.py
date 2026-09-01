"""Geometrically aligned PIL image/index-mask transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def _to_tensors(
    image: Image.Image,
    mask: Image.Image,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[Tensor, Tensor]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).div_(255.0)
    mean_tensor = image_tensor.new_tensor(mean)[:, None, None]
    std_tensor = image_tensor.new_tensor(std)[:, None, None]
    image_tensor = (image_tensor - mean_tensor) / std_tensor
    mask_array = np.asarray(mask, dtype=np.int64).copy()
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a single-channel indexed mask, found shape {mask_array.shape}.")
    return image_tensor, torch.from_numpy(mask_array)


def _random_crop_parameters(
    image: Image.Image,
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> tuple[int, int, int, int]:
    width, height = image.size
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio, dtype=torch.float64))
    log_ratio_min, log_ratio_max = (float(item) for item in log_ratio)
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(*scale).item())
        aspect_ratio = math.exp(
            float(torch.empty(1).uniform_(log_ratio_min, log_ratio_max).item())
        )
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_width + 1, (1,)).item())
            return top, left, crop_height, crop_width
    input_ratio = width / height
    if input_ratio < ratio[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio[0]))
    elif input_ratio > ratio[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio[1]))
    else:
        crop_width, crop_height = width, height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width


@dataclass(frozen=True, slots=True)
class PairedSegmentationTransform:
    """Apply exactly the same geometry to an image and its indexed mask."""

    height: int
    width: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    training: bool
    scale: tuple[float, float] = (1.0, 1.0)
    ratio: tuple[float, float] = (1.0, 1.0)
    horizontal_flip_probability: float = 0.0

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: image={image.size}, mask={mask.size}.")
        if self.training:
            top, left, crop_height, crop_width = _random_crop_parameters(
                image,
                scale=self.scale,
                ratio=self.ratio,
            )
            box = (left, top, left + crop_width, top + crop_height)
            image = image.crop(box)
            mask = mask.crop(box)
        image = image.resize((self.width, self.height), resample=Image.Resampling.BILINEAR)
        # Indexed semantic masks must never use a continuous interpolation mode.
        mask = mask.resize((self.width, self.height), resample=Image.Resampling.NEAREST)
        if self.training and float(torch.rand(1).item()) < self.horizontal_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return _to_tensors(image, mask, mean=self.mean, std=self.std)
