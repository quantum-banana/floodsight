"""Offline synthetic forward/loss/backward/metrics/checkpoint smoke test."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .checkpoint import TrainingState, load_checkpoint, save_checkpoint
from .checkpoint_probe import run_fresh_process_checkpoint_probe
from .metrics import SegmentationMetrics
from .model import build_tiny_offline_segformer, logits_at_label_resolution
from .optim import build_scheduler
from .reproducibility import make_generator, seed_everything
from .supervision import PartialCrossEntropyLoss, build_class_availability

SYNTHETIC_CONFIG_SHA256 = "0" * 64
SYNTHETIC_MANIFEST_SHA256 = {"synthetic://segmentation": "1" * 64}
SYNTHETIC_MANIFEST_FINGERPRINT = {"synthetic://segmentation": "2" * 64}
SYNTHETIC_TAXONOMY_SHA256 = {"synthetic://taxonomy": "3" * 64}
SYNTHETIC_INPUT_PROVENANCE = {"mode": "DEMO_SIMULATED"}
SYNTHETIC_SUPPORT = {
    "floodnet": frozenset({0, 1, 2, 3, 6, 7, 12, 13, 14, 15}),
    "rescuenet": frozenset({0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 15}),
}


class _SyntheticSchedulerConfig:
    warmup_ratio = 0.0
    power = 1.0


def run_synthetic_smoke(output_dir: Path, *, device_name: str = "cpu") -> dict[str, Any]:
    """Exercise the ML path using generated tensors only; never open a dataset."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for smoke testing but is unavailable.")
    seed_everything(1234, deterministic_algorithms=True, cudnn_benchmark=False)
    data_generator = make_generator(5678)
    model = build_tiny_offline_segformer(num_labels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, _SyntheticSchedulerConfig(), total_steps=2)  # type: ignore[arg-type]
    pixel_values = torch.randn((2, 3, 32, 32), device=device)
    labels = torch.zeros((2, 32, 32), dtype=torch.long, device=device)
    labels[0, :16, :16] = 3
    labels[0, 16:, 16:] = 6
    labels[1, :16, :16] = 4
    labels[1, 16:, 16:] = 9
    labels[:, :2, :] = 255
    availability = build_class_availability(
        ["floodnet", "rescuenet"],
        supported_class_ids=SYNTHETIC_SUPPORT,
        num_labels=16,
        device=device,
    )
    loss_function = PartialCrossEntropyLoss(class_weights=(1.0,) * 16, ignore_index=255)
    model.train()
    outputs = model(pixel_values=pixel_values)
    logits = logits_at_label_resolution(outputs.logits, labels)
    loss = loss_function(logits, labels, availability)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Synthetic SegFormer loss is non-finite.")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    if not math.isfinite(float(gradient_norm)):
        raise RuntimeError("Synthetic SegFormer gradients are non-finite.")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    metrics = SegmentationMetrics(num_labels=16, ignore_index=255)
    metrics.update(logits.detach(), labels, availability)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "synthetic-smoke.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(epoch=1, global_step=1, best_metric=0.0),
        config_sha256=SYNTHETIC_CONFIG_SHA256,
        manifest_sha256=SYNTHETIC_MANIFEST_SHA256,
        manifest_fingerprint=SYNTHETIC_MANIFEST_FINGERPRINT,
        taxonomy_sha256=SYNTHETIC_TAXONOMY_SHA256,
        input_provenance=SYNTHETIC_INPUT_PROVENANCE,
        run_directory=output_dir,
        data_generator=data_generator,
        provenance="DEMO_SIMULATED",
    )
    expected_next_random = torch.rand(4)
    _ = torch.rand(64)
    resumed_state = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_config_sha256=SYNTHETIC_CONFIG_SHA256,
        expected_manifest_sha256=SYNTHETIC_MANIFEST_SHA256,
        expected_manifest_fingerprint=SYNTHETIC_MANIFEST_FINGERPRINT,
        expected_taxonomy_sha256=SYNTHETIC_TAXONOMY_SHA256,
        expected_input_provenance=SYNTHETIC_INPUT_PROVENANCE,
        expected_run_directory=output_dir,
        data_generator=data_generator,
        map_location=device,
        expected_provenance="DEMO_SIMULATED",
    )
    resumed_next_random = torch.rand(4)
    rng_match = bool(torch.equal(expected_next_random, resumed_next_random))
    if not rng_match or resumed_state != TrainingState(epoch=1, global_step=1, best_metric=0.0):
        raise RuntimeError("Synthetic checkpoint did not restore exact state and RNG.")
    fresh_process_probe = run_fresh_process_checkpoint_probe(
        output_dir / "fresh-process-checkpoint-probe"
    )
    report_path = output_dir / "synthetic-smoke-report.json"
    report = {
        "status": "PASS",
        "provenance": "DEMO_SIMULATED",
        "real_dataset_access": False,
        "real_training": False,
        "offline_model": True,
        "forward": "PASS",
        "loss": float(loss.detach()),
        "loss_finite": math.isfinite(float(loss.detach())),
        "backward": "PASS",
        "gradient_norm": float(gradient_norm),
        "optimizer_step": "SYNTHETIC_ONLY",
        "metrics": metrics.compute(),
        "checkpoint_reload": "PASS",
        "rng_resume_match": rng_match,
        "fresh_process_checkpoint_rng_continuation": fresh_process_probe,
        "checkpoint": str(checkpoint_path),
        "report_path": str(report_path),
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report
