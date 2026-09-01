"""Crash-safe, content-bound checkpoint generations for Ultralytics training."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256

CHECKPOINT_DIRECTORY = ".floodsight-checkpoints"
CHECKPOINT_POINTER = ".floodsight-last-checkpoint.json"
CHECKPOINT_SCHEMA = "floodsight-detection-checkpoint-v1"
POINTER_SCHEMA = "floodsight-detection-checkpoint-pointer-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class TrustedCheckpoint:
    path: Path
    sha256: str
    metadata_path: Path
    metadata_sha256: str
    epoch: int
    rng_state: dict[str, Any]
    trainer_arguments: dict[str, Any]


def _fail(message: str, code: str = "checkpoint_integrity_failed") -> None:
    raise DetectionInfrastructureError(message, code=code)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        result = path.lstat()
    except OSError as exc:
        raise DetectionInfrastructureError(
            f"Missing {label}: {path}", code="checkpoint_integrity_failed"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(result.st_mode):
        _fail(f"Unsafe {label}: {path}")
    return result


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o440) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            f"Refusing to overwrite checkpoint evidence: {path}",
            code="checkpoint_collision",
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                _fail(f"Short write while persisting checkpoint evidence: {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            os.close(descriptor)
        raise
    os.close(descriptor)
    _fsync_directory(path.parent)


def write_json_exclusive(path: Path, payload: dict[str, Any], *, mode: int = 0o440) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_exclusive(path, encoded, mode=mode)


def _replace_bytes_atomic(path: Path, payload: bytes, *, mode: int) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{random.randrange(1 << 32):08x}"
    _write_bytes_exclusive(temporary, payload, mode=mode)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _encode_torch_state(state: Any) -> str:
    return base64.b64encode(bytes(state.detach().cpu().tolist())).decode("ascii")


def _decode_torch_state(encoded: str, torch: Any) -> Any:
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, TypeError) as exc:
        raise DetectionInfrastructureError(
            "Checkpoint RNG state contains invalid base64.",
            code="checkpoint_rng_invalid",
        ) from exc
    return torch.tensor(list(payload), dtype=torch.uint8)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def capture_rng_state(trainer: Any | None = None) -> dict[str, Any]:
    """Capture process and loader RNG state after a safely persisted epoch."""

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    state: dict[str, Any] = {
        "schema_version": "floodsight-rng-state-v1",
        "python": list(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].astype("uint32", copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": _encode_torch_state(torch.get_rng_state()),
        "torch_cuda": (
            [_encode_torch_state(item) for item in torch.cuda.get_rng_state_all()]
            if cuda_initialized
            else []
        ),
        "torch_cuda_initialized": cuda_initialized,
        "torch_cuda_device_count": (
            int(torch.cuda.device_count()) if cuda_initialized else 0
        ),
    }
    if trainer is not None:
        loader = getattr(trainer, "train_loader", None)
        loader_generator = getattr(loader, "generator", None)
        sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
        state["train_loader_generator"] = (
            _encode_torch_state(loader_generator.get_state())
            if loader_generator is not None
            else None
        )
        state["train_sampler_generator"] = (
            _encode_torch_state(sampler_generator.get_state())
            if sampler_generator is not None and sampler_generator is not loader_generator
            else None
        )
    state["state_sha256"] = stable_sha256(state)
    return state


def restore_rng_state(state: dict[str, Any], trainer: Any | None = None) -> None:
    """Restore every captured process and DataLoader RNG, failing on drift."""

    import numpy as np
    import torch

    if not isinstance(state, dict) or state.get("schema_version") != "floodsight-rng-state-v1":
        _fail("Checkpoint RNG state schema is invalid.", "checkpoint_rng_invalid")
    declared_hash = state.get("state_sha256")
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    if (
        not isinstance(declared_hash, str)
        or _HEX64.fullmatch(declared_hash) is None
        or stable_sha256(unsigned) != declared_hash
    ):
        _fail("Checkpoint RNG state failed its integrity check.", "checkpoint_rng_invalid")
    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, dict):
        _fail("Checkpoint NumPy RNG state is invalid.", "checkpoint_rng_invalid")
    try:
        random.setstate(_nested_tuple(state["python"]))
        np.random.set_state(
            (
                numpy_state["algorithm"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(_decode_torch_state(state["torch_cpu"], torch))
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, list):
            raise TypeError("torch_cuda is not a list")
        cuda_was_initialized = state.get("torch_cuda_initialized")
        if not isinstance(cuda_was_initialized, bool):
            raise TypeError("torch_cuda_initialized is not a boolean")
        if cuda_states:
            if (
                not cuda_was_initialized
                or not torch.cuda.is_available()
                or len(cuda_states) != torch.cuda.device_count()
            ):
                _fail("Checkpoint CUDA RNG device topology drifted.", "checkpoint_rng_invalid")
            torch.cuda.set_rng_state_all(
                [_decode_torch_state(encoded, torch) for encoded in cuda_states]
            )
        if trainer is not None:
            loader = getattr(trainer, "train_loader", None)
            loader_generator = getattr(loader, "generator", None)
            sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
            loader_state = state.get("train_loader_generator")
            sampler_state = state.get("train_sampler_generator")
            if loader_state is not None:
                if loader_generator is None:
                    _fail(
                        "Resume loader omitted its bound RNG generator.", "checkpoint_rng_invalid"
                    )
                loader_generator.set_state(_decode_torch_state(loader_state, torch))
            if sampler_state is not None:
                if sampler_generator is None:
                    _fail(
                        "Resume sampler omitted its bound RNG generator.", "checkpoint_rng_invalid"
                    )
                sampler_generator.set_state(_decode_torch_state(sampler_state, torch))
    except DetectionInfrastructureError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise DetectionInfrastructureError(
            "Checkpoint RNG state could not be restored.",
            code="checkpoint_rng_invalid",
        ) from exc


def publish_checkpoint_generation(
    *,
    run_directory: Path,
    run_metadata_path: Path,
    live_checkpoint: Path,
    epoch: int,
    trainer_arguments: dict[str, Any],
    data_yaml: Path,
    trainer: Any,
) -> TrustedCheckpoint:
    """Copy an Ultralytics last.pt into a durable immutable generation and point to it."""

    if epoch < 0:
        _fail("Checkpoint epoch is invalid.")
    _safe_regular(run_metadata_path, label="run reservation metadata")
    try:
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Run reservation metadata is unreadable.", code="checkpoint_integrity_failed"
        ) from exc
    reservation_sha256 = run_metadata.get("reservation_sha256")
    if not isinstance(reservation_sha256, str) or _HEX64.fullmatch(reservation_sha256) is None:
        _fail("Run reservation identity is unavailable.")
    live_stat = _safe_regular(live_checkpoint, label="Ultralytics live checkpoint")
    if live_stat.st_size < 1:
        _fail("Ultralytics produced an empty checkpoint.")
    data_yaml_stat = _safe_regular(data_yaml, label="frozen dataset YAML")
    if data_yaml_stat.st_size < 1:
        _fail("Frozen dataset YAML is empty.")
    checkpoint_sha256 = sha256_file(live_checkpoint)
    generation_root = run_directory / CHECKPOINT_DIRECTORY
    if generation_root.is_symlink() or (generation_root.exists() and not generation_root.is_dir()):
        _fail("Checkpoint generation directory is unsafe.")
    generation_root.mkdir(mode=0o750, exist_ok=True)
    basename = f"epoch-{epoch:06d}-{checkpoint_sha256}.pt"
    generation_path = generation_root / basename
    temporary = generation_root / f".{basename}.tmp-{os.getpid()}"
    if generation_path.exists() or generation_path.is_symlink() or temporary.exists():
        raise DetectionInfrastructureError(
            "Checkpoint generation already exists.", code="checkpoint_collision"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o440)
    try:
        with (
            live_checkpoint.open("rb") as source,
            os.fdopen(descriptor, "wb", closefd=False) as target,
        ):
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.fchmod(descriptor, 0o440)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    if sha256_file(temporary) != checkpoint_sha256:
        temporary.unlink(missing_ok=True)
        _fail("Checkpoint changed while its immutable generation was copied.")
    try:
        os.link(temporary, generation_path)
        _fsync_directory(generation_root)
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            "Checkpoint generation collision.", code="checkpoint_collision"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    rng_state = capture_rng_state(trainer)
    metadata_body = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_relative_path": generation_path.relative_to(run_directory).as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": generation_path.stat().st_size,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "reservation_sha256": reservation_sha256,
        "run_metadata_sha256": sha256_file(run_metadata_path),
        "data_yaml_relative_path": data_yaml.relative_to(run_directory).as_posix(),
        "data_yaml_sha256": sha256_file(data_yaml),
        "trainer_arguments": trainer_arguments,
        "trainer_arguments_sha256": stable_sha256(trainer_arguments),
        "rng_state": rng_state,
        "rng_state_sha256": rng_state["state_sha256"],
    }
    metadata = {**metadata_body, "metadata_sha256": stable_sha256(metadata_body)}
    metadata_path = generation_path.with_suffix(".json")
    write_json_exclusive(metadata_path, metadata)
    pointer_body = {
        "schema_version": POINTER_SCHEMA,
        "metadata_relative_path": metadata_path.relative_to(run_directory).as_posix(),
        "metadata_sha256": metadata["metadata_sha256"],
        "checkpoint_relative_path": metadata_body["checkpoint_relative_path"],
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": epoch,
        "reservation_sha256": reservation_sha256,
    }
    pointer = {**pointer_body, "pointer_sha256": stable_sha256(pointer_body)}
    pointer_path = run_directory / CHECKPOINT_POINTER
    _replace_bytes_atomic(
        pointer_path,
        (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o440,
    )
    return TrustedCheckpoint(
        path=generation_path,
        sha256=checkpoint_sha256,
        metadata_path=metadata_path,
        metadata_sha256=metadata["metadata_sha256"],
        epoch=epoch,
        rng_state=rng_state,
        trainer_arguments=trainer_arguments,
    )


def load_trusted_checkpoint(
    *,
    run_directory: Path,
    run_metadata_path: Path,
    declared_live_checkpoint: Path,
    expected_trainer_arguments: dict[str, Any],
) -> TrustedCheckpoint:
    """Validate the live selector, durable pointer, metadata, and immutable bytes."""

    live_stat = _safe_regular(declared_live_checkpoint, label="declared weights/last.pt")
    if live_stat.st_size < 1:
        _fail("Declared resume checkpoint is empty or torn.")
    pointer_path = run_directory / CHECKPOINT_POINTER
    _safe_regular(pointer_path, label="checkpoint pointer")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Checkpoint pointer is unreadable.", code="checkpoint_integrity_failed"
        ) from exc
    pointer_fields = {
        "schema_version",
        "metadata_relative_path",
        "metadata_sha256",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "epoch",
        "reservation_sha256",
        "pointer_sha256",
    }
    if not isinstance(pointer, dict) or set(pointer) != pointer_fields:
        _fail("Checkpoint pointer fields drifted.")
    pointer_unsigned = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
    if (
        pointer.get("schema_version") != POINTER_SCHEMA
        or not isinstance(pointer.get("pointer_sha256"), str)
        or stable_sha256(pointer_unsigned) != pointer["pointer_sha256"]
    ):
        _fail("Checkpoint pointer failed its integrity check.")
    try:
        metadata_path = (run_directory / pointer["metadata_relative_path"]).resolve(strict=True)
        checkpoint_path = (run_directory / pointer["checkpoint_relative_path"]).resolve(strict=True)
        metadata_path.relative_to(run_directory / CHECKPOINT_DIRECTORY)
        checkpoint_path.relative_to(run_directory / CHECKPOINT_DIRECTORY)
    except (OSError, TypeError, ValueError) as exc:
        raise DetectionInfrastructureError(
            "Checkpoint pointer escapes its trusted generation directory.",
            code="checkpoint_integrity_failed",
        ) from exc
    _safe_regular(metadata_path, label="checkpoint metadata")
    checkpoint_stat = _safe_regular(checkpoint_path, label="immutable checkpoint generation")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Checkpoint or run metadata is unreadable.", code="checkpoint_integrity_failed"
        ) from exc
    metadata_unsigned = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    expected_metadata_fields = {
        "schema_version",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "epoch",
        "next_epoch",
        "reservation_sha256",
        "run_metadata_sha256",
        "data_yaml_relative_path",
        "data_yaml_sha256",
        "trainer_arguments",
        "trainer_arguments_sha256",
        "rng_state",
        "rng_state_sha256",
        "metadata_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_fields:
        _fail("Checkpoint metadata fields drifted.")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    live_sha256 = sha256_file(declared_live_checkpoint)
    reservation_sha256 = run_metadata.get("reservation_sha256")
    try:
        data_yaml = (run_directory / metadata["data_yaml_relative_path"]).resolve(strict=True)
        data_yaml.relative_to(run_directory / "dataset-contract")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise DetectionInfrastructureError(
            "Checkpoint dataset YAML escapes its frozen contract directory.",
            code="checkpoint_integrity_failed",
        ) from exc
    _safe_regular(data_yaml, label="checkpoint-bound dataset YAML")
    epoch = metadata.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        _fail("Checkpoint metadata epoch is invalid.")
    conditions = (
        metadata.get("schema_version") == CHECKPOINT_SCHEMA,
        metadata.get("metadata_sha256") == stable_sha256(metadata_unsigned),
        metadata.get("metadata_sha256") == pointer.get("metadata_sha256"),
        metadata.get("checkpoint_relative_path") == pointer.get("checkpoint_relative_path"),
        metadata.get("checkpoint_sha256") == pointer.get("checkpoint_sha256"),
        metadata.get("checkpoint_sha256") == checkpoint_sha256 == live_sha256,
        metadata.get("checkpoint_size_bytes") == checkpoint_stat.st_size == live_stat.st_size,
        metadata.get("reservation_sha256")
        == pointer.get("reservation_sha256")
        == reservation_sha256,
        metadata.get("run_metadata_sha256") == sha256_file(run_metadata_path),
        metadata.get("data_yaml_sha256") == sha256_file(data_yaml),
        metadata.get("trainer_arguments") == expected_trainer_arguments,
        metadata.get("trainer_arguments_sha256") == stable_sha256(expected_trainer_arguments),
        metadata.get("rng_state_sha256") == metadata.get("rng_state", {}).get("state_sha256"),
        metadata.get("next_epoch") == epoch + 1,
        pointer.get("epoch") == epoch,
    )
    if not all(conditions):
        _fail("Checkpoint bytes, sidecar, live selector, or run identity drifted.")
    # Validate RNG structure/digest without mutating the caller's process.
    rng_state = metadata["rng_state"]
    if stable_sha256(
        {key: value for key, value in rng_state.items() if key != "state_sha256"}
    ) != rng_state.get("state_sha256"):
        _fail("Checkpoint RNG state metadata drifted.", "checkpoint_rng_invalid")
    return TrustedCheckpoint(
        path=checkpoint_path,
        sha256=checkpoint_sha256,
        metadata_path=metadata_path,
        metadata_sha256=metadata["metadata_sha256"],
        epoch=epoch,
        rng_state=rng_state,
        trainer_arguments=metadata["trainer_arguments"],
    )


def restore_live_checkpoint(trusted: TrustedCheckpoint, live_checkpoint: Path) -> None:
    """Atomically replace Ultralytics' stripped last.pt with the trusted resumable bytes."""

    _safe_regular(trusted.path, label="trusted checkpoint generation")
    payload = trusted.path.read_bytes()
    if not payload:
        _fail("Trusted checkpoint generation is empty.")
    if sha256_file(trusted.path) != trusted.sha256:
        _fail("Trusted checkpoint generation drifted before live restoration.")
    if live_checkpoint.is_symlink() or (live_checkpoint.exists() and not live_checkpoint.is_file()):
        _fail("Ultralytics live checkpoint path became unsafe.")
    _replace_bytes_atomic(live_checkpoint, payload, mode=0o640)
    if sha256_file(live_checkpoint) != trusted.sha256:
        _fail("Live checkpoint restoration failed its hash check.")
