"""Optimizer parameter grouping and warmup-polynomial scheduling."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import OptimizerConfig, SchedulerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build AdamW with explicit decay and decoder learning-rate groups."""

    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_decoder = name.startswith("decode_head") or ".decode_head." in name
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        groups[(is_decoder, no_decay)].append(parameter)
    parameter_groups = []
    for (is_decoder, no_decay), parameters in groups.items():
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": config.learning_rate
                * (config.decoder_learning_rate_multiplier if is_decoder else 1.0),
                "weight_decay": 0.0 if no_decay else config.weight_decay,
            }
        )
    return torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
        amsgrad=config.amsgrad,
        maximize=config.maximize,
        foreach=config.foreach,
        capturable=config.capturable,
        differentiable=config.differentiable,
        fused=config.fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a step-based linear warmup followed by polynomial decay."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    warmup_steps = math.floor(total_steps * config.warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return (remaining / decay_steps) ** config.power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
