"""Synthetic two-process checkpoint/RNG continuation regression probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifact import ModelArtifactSpec, validate_model_artifact
from .checkpoint import TrainingState, load_checkpoint, save_checkpoint
from .config import load_config
from .integrity import training_source_sha256
from .model import build_segformer
from .optim import build_optimizer, build_scheduler
from .reproducibility import make_generator, seed_everything
from .runtime import require_h100, validate_runtime_versions

_CONFIG_SHA256 = "4" * 64
_MANIFEST_SHA256 = {"synthetic://fresh-process": "5" * 64}
_MANIFEST_FINGERPRINT = {"synthetic://fresh-process": "6" * 64}
_TAXONOMY_SHA256 = {"synthetic://taxonomy": "7" * 64}
_INPUT_PROVENANCE = {"mode": "DEMO_SIMULATED_FRESH_PROCESS"}


def _components(seed: int) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.Generator,
]:
    torch.set_num_threads(1)
    seed_everything(seed, deterministic_algorithms=True, cudnn_benchmark=False)
    generator = make_generator(seed + 1)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - 0.1 * step),
    )
    return model, optimizer, scheduler, generator


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
) -> float:
    inputs = torch.rand((4, 3), generator=generator)
    targets = torch.rand((4, 2))
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def _continuation(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
) -> dict[str, Any]:
    draws = {
        "python": random.random(),
        "numpy": np.random.random(4).tolist(),
        "torch": torch.rand(4).tolist(),
        "data_generator": torch.rand(4, generator=generator).tolist(),
    }
    loss = _optimizer_step(model, optimizer, scheduler, generator)
    parameters = {
        name: tensor.detach().cpu().tolist()
        for name, tensor in sorted(model.state_dict().items())
    }
    optimizer_steps = sorted(
        int(state["step"].item())
        for state in optimizer.state.values()
        if isinstance(state, dict) and isinstance(state.get("step"), torch.Tensor)
    )
    return {
        "draws": draws,
        "continuation_loss": loss,
        "model_state": parameters,
        "optimizer_steps": optimizer_steps,
        "scheduler_last_epoch": scheduler.last_epoch,
        "learning_rates": scheduler.get_last_lr(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_fingerprint(value: Any) -> str:
    """Hash nested Torch state without pickle, device, or object-address dependence."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            ordered_keys = sorted(
                item,
                key=lambda candidate: (type(candidate).__name__, repr(candidate)),
            )
            for key in ordered_keys:
                update(key)
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"list\0" if isinstance(item, list) else b"tuple\0")
            for nested in item:
                update(nested)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(item, allow_nan=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\0")
            return
        raise TypeError(f"Unsupported checkpoint-state value for fingerprint: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _rng_continuation(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    draws: dict[str, Any] = {
        "python": random.random(),
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4).tolist(),
        "data_generator": torch.rand(4, generator=generator).tolist(),
    }
    if device.type == "cuda":
        draws["torch_cuda"] = torch.rand(4, device=device).cpu().tolist()
    return draws


def _component_fingerprints(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
) -> dict[str, str]:
    return {
        "model": state_fingerprint(model.state_dict()),
        "optimizer": state_fingerprint(optimizer.state_dict()),
        "scheduler": state_fingerprint(scheduler.state_dict()),
        "scaler": state_fingerprint(scaler.state_dict() if scaler is not None else None),
    }


def _production_resume(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise RuntimeError("Production checkpoint request must be a JSON object.")
    validate_runtime_versions()
    config = load_config(Path(request["config_path"]))
    if config.sha256 != request["config_sha256"]:
        raise RuntimeError("Production checkpoint probe configuration hash drifted.")
    child_source_sha256 = training_source_sha256()
    if child_source_sha256 != request["training_source_sha256"]:
        raise RuntimeError("Production checkpoint probe executable source fingerprint drifted.")
    if request["input_provenance"].get("training_source_sha256") != child_source_sha256:
        raise RuntimeError("Production checkpoint input provenance has stale executable source.")
    artifact_raw = request["model_artifact"]
    artifact_spec = ModelArtifactSpec(
        safetensors_path=Path(artifact_raw["safetensors_path"]),
        safetensors_sha256=artifact_raw["safetensors_sha256"],
        provenance_path=Path(artifact_raw["provenance_path"]),
        provenance_sha256=artifact_raw["provenance_sha256"],
    )
    artifact = validate_model_artifact(artifact_spec, config.model)
    device = torch.device(request["device"])
    require_h100(device)
    seed_everything(
        config.reproducibility.seed,
        deterministic_algorithms=config.reproducibility.deterministic_algorithms,
        cudnn_benchmark=config.reproducibility.cudnn_benchmark,
    )
    generator = make_generator(config.reproducibility.seed)
    model = build_segformer(config.model, artifact).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=request["total_steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = load_checkpoint(
        Path(request["checkpoint_path"]),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_sha256=config.sha256,
        expected_manifest_sha256=request["manifest_sha256"],
        expected_manifest_fingerprint=request["manifest_fingerprint"],
        expected_taxonomy_sha256=request["taxonomy_sha256"],
        expected_input_provenance=request["input_provenance"],
        expected_authorization_provenance={},
        expected_run_directory=Path(request["run_directory"]),
        data_generator=generator,
        map_location=device,
        expected_provenance="REAL_ML_OUTPUT",
    )
    actual = {
        "resumer_pid": os.getpid(),
        "training_state": asdict(state),
        "component_fingerprints": _component_fingerprints(
            model,
            optimizer,
            scheduler,
            scaler,
        ),
        "rng_continuation": _rng_continuation(generator, device),
        "training_source_sha256": child_source_sha256,
    }
    _write_json(request_path.parent / "actual.json", actual)


def run_fresh_process_production_checkpoint_probe(
    output_directory: Path,
    *,
    checkpoint_path: Path,
    config_path: Path,
    model_artifact_spec: ModelArtifactSpec,
    training_state: TrainingState,
    manifest_sha256: Mapping[str, str],
    manifest_fingerprint: Mapping[str, str],
    taxonomy_sha256: Mapping[str, str],
    input_provenance: Mapping[str, str],
    run_directory: Path,
    data_generator: torch.Generator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Reload the actual SegFormer smoke checkpoint in a new H100 Python process."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    normalized_artifact = model_artifact_spec.normalized()
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    expected = {
        "creator_pid": os.getpid(),
        "training_source_sha256": training_source_sha256(),
        "training_state": asdict(training_state),
        "component_fingerprints": _component_fingerprints(
            model,
            optimizer,
            scheduler,
            scaler,
        ),
        "rng_continuation": _rng_continuation(data_generator, device),
    }
    request = {
        "config_path": str(config_path.expanduser().resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "run_directory": str(run_directory.expanduser().resolve()),
        "device": f"cuda:{device_index}",
        "total_steps": 1,
        "manifest_sha256": dict(sorted(manifest_sha256.items())),
        "manifest_fingerprint": dict(sorted(manifest_fingerprint.items())),
        "taxonomy_sha256": dict(sorted(taxonomy_sha256.items())),
        "input_provenance": dict(sorted(input_provenance.items())),
        "training_source_sha256": expected["training_source_sha256"],
        "model_artifact": {
            "safetensors_path": str(normalized_artifact.safetensors_path),
            "safetensors_sha256": normalized_artifact.safetensors_sha256,
            "provenance_path": str(normalized_artifact.provenance_path),
            "provenance_sha256": normalized_artifact.provenance_sha256,
        },
    }
    request_path = output / "request.json"
    _write_json(request_path, request)
    _write_json(output / "expected.json", expected)
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(package_root)
        if not prior_pythonpath
        else f"{package_root}{os.pathsep}{prior_pythonpath}"
    )
    command = [
        sys.executable,
        "-m",
        "floodsight_segmentation.checkpoint_probe",
        "--phase",
        "production-resume",
        "--request",
        str(request_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Fresh-process production checkpoint reload failed: {detail}")
    actual = json.loads((output / "actual.json").read_text(encoding="utf-8"))
    if actual.get("resumer_pid") == expected["creator_pid"]:
        raise RuntimeError("Production checkpoint probe did not use a fresh Python process.")
    for field in (
        "training_state",
        "component_fingerprints",
        "rng_continuation",
        "training_source_sha256",
    ):
        if actual.get(field) != expected[field]:
            raise RuntimeError(f"Fresh-process production checkpoint mismatch at {field}.")
    report = {
        "status": "PASS",
        "provenance": "REAL_ML_OUTPUT",
        "fresh_python_process": True,
        "creator_pid": expected["creator_pid"],
        "resumer_pid": actual["resumer_pid"],
        "training_state": "PASS",
        "model_state": "PASS",
        "optimizer_state": "PASS",
        "scheduler_state": "PASS",
        "scaler_state": "PASS",
        "python_numpy_torch_cpu_cuda_generator_rng": "PASS",
        "child_executable_source_rehash": "PASS",
        "training_source_sha256": actual["training_source_sha256"],
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "request": str(request_path),
    }
    _write_json(output / "probe-report.json", report)
    return report


def _create(directory: Path) -> None:
    model, optimizer, scheduler, generator = _components(137)
    _optimizer_step(model, optimizer, scheduler, generator)
    checkpoint = directory / "last.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(epoch=1, global_step=1, best_metric=0.25),
        config_sha256=_CONFIG_SHA256,
        manifest_sha256=_MANIFEST_SHA256,
        manifest_fingerprint=_MANIFEST_FINGERPRINT,
        taxonomy_sha256=_TAXONOMY_SHA256,
        input_provenance=_INPUT_PROVENANCE,
        run_directory=directory,
        data_generator=generator,
        provenance="DEMO_SIMULATED",
    )
    _write_json(
        directory / "expected.json",
        {
            "creator_pid": os.getpid(),
            "continuation": _continuation(model, optimizer, scheduler, generator),
        },
    )


def _resume(directory: Path) -> None:
    expected = json.loads((directory / "expected.json").read_text(encoding="utf-8"))
    model, optimizer, scheduler, generator = _components(991)
    state = load_checkpoint(
        directory / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_config_sha256=_CONFIG_SHA256,
        expected_manifest_sha256=_MANIFEST_SHA256,
        expected_manifest_fingerprint=_MANIFEST_FINGERPRINT,
        expected_taxonomy_sha256=_TAXONOMY_SHA256,
        expected_input_provenance=_INPUT_PROVENANCE,
        expected_run_directory=directory,
        data_generator=generator,
        map_location="cpu",
        expected_provenance="DEMO_SIMULATED",
    )
    actual = _continuation(model, optimizer, scheduler, generator)
    if state != TrainingState(epoch=1, global_step=1, best_metric=0.25):
        raise RuntimeError(f"Fresh-process checkpoint state mismatch: {state!r}")
    if actual != expected.get("continuation"):
        raise RuntimeError("Fresh-process RNG/model/optimizer continuation does not match.")
    _write_json(
        directory / "actual.json",
        {
            "resumer_pid": os.getpid(),
            "training_state": {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
            },
            "continuation": actual,
        },
    )


def run_fresh_process_checkpoint_probe(output_directory: Path) -> dict[str, Any]:
    """Create and resume a checkpoint in two independent Python interpreters."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(package_root)
        if not prior_pythonpath
        else f"{package_root}{os.pathsep}{prior_pythonpath}"
    )
    commands = []
    for phase in ("create", "resume"):
        command = [
            sys.executable,
            "-m",
            "floodsight_segmentation.checkpoint_probe",
            "--phase",
            phase,
            "--directory",
            str(output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        commands.append({"phase": phase, "returncode": completed.returncode})
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"Fresh-process checkpoint phase {phase!r} failed: {detail}")
    expected = json.loads((output / "expected.json").read_text(encoding="utf-8"))
    actual = json.loads((output / "actual.json").read_text(encoding="utf-8"))
    if expected["creator_pid"] == actual["resumer_pid"]:
        raise RuntimeError("Checkpoint probe did not use distinct Python processes.")
    report = {
        "status": "PASS",
        "provenance": "DEMO_SIMULATED",
        "fresh_python_processes": True,
        "creator_pid": expected["creator_pid"],
        "resumer_pid": actual["resumer_pid"],
        "checkpoint_state": "PASS",
        "python_numpy_torch_generator_rng": "PASS",
        "model_optimizer_scheduler_continuation": "PASS",
        "commands": commands,
        "checkpoint": str(output / "last.pt"),
    }
    _write_json(output / "probe-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("create", "resume", "production-resume"),
        required=True,
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "production-resume":
        if args.request is None or args.directory is not None:
            parser.error("production-resume requires --request and forbids --directory")
        _production_resume(args.request.expanduser().resolve())
        return 0
    if args.directory is None or args.request is not None:
        parser.error("create/resume require --directory and forbid --request")
    directory = args.directory.expanduser().resolve()
    if args.phase == "create":
        _create(directory)
    else:
        _resume(directory)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
