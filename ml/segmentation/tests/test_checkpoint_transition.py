"""Focused contract tests for the one-time prepared-data source transition."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from floodsight_segmentation.checkpoint import (
    TrainingState,
    load_checkpoint_transition,
    save_checkpoint,
)
from floodsight_segmentation.errors import CheckpointError
from floodsight_segmentation.reproducibility import make_generator, seed_everything
from floodsight_segmentation.transition import ALLOWED_TRANSITION, TRANSITION_SCHEMA


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization(seed: str) -> dict[str, str]:
    return {
        "approval_record_sha256": seed * 64,
        "human_review_sha256": chr(ord(seed) + 1) * 64,
        "real_smoke_report_sha256": chr(ord(seed) + 2) * 64,
    }


def _write_record(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return _digest(path)


def _case(tmp_path: Path) -> dict[str, Any]:
    seed_everything(101, deterministic_algorithms=True, cudnn_benchmark=False)
    generator = make_generator(102)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config_sha256 = "a" * 64
    manifests = {"manifest": "b" * 64}
    fingerprints = {"manifest": "c" * 64}
    taxonomy = {"taxonomy": "d" * 64}
    predecessor_input = {
        "model_revision": "immutable-model",
        "training_source_sha256": "e" * 64,
        "runtime_fingerprint": "f" * 64,
    }
    successor_input = {
        **predecessor_input,
        "training_source_sha256": "1" * 64,
        "prepared_fast_path_record_path": str((tmp_path / "prepared.json").resolve()),
        "prepared_fast_path_record_sha256": "2" * 64,
        "prepared_fast_path_snapshot_fingerprint": "3" * 64,
        "prepared_fast_path_loader_threads": "4",
        "torch_cpu_threads": "2",
    }
    predecessor_authorization = _authorization("4")
    successor_authorization = {
        "approval_record_sha256": "7" * 64,
        "human_review_sha256": predecessor_authorization["human_review_sha256"],
        "real_smoke_report_sha256": "8" * 64,
    }
    checkpoint = tmp_path / "last.pt"
    state = TrainingState(epoch=1, global_step=293, best_metric=0.4364086588782282)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=state,
        config_sha256=config_sha256,
        manifest_sha256=manifests,
        manifest_fingerprint=fingerprints,
        taxonomy_sha256=taxonomy,
        input_provenance=predecessor_input,
        authorization_provenance=predecessor_authorization,
        run_directory=tmp_path,
        data_generator=generator,
        provenance="REAL_ML_OUTPUT",
    )
    payload = {
        "schema_version": TRANSITION_SCHEMA,
        "allowed_change": ALLOWED_TRANSITION,
        "predecessor_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _digest(checkpoint),
        },
        "frozen_run": {
            "config_sha256": config_sha256,
            "manifest_sha256": manifests,
            "manifest_fingerprint": fingerprints,
            "taxonomy_sha256": taxonomy,
            "run_directory": str(tmp_path.resolve()),
        },
        "training_state": {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
        },
        "predecessor": {
            "input_provenance": predecessor_input,
            "authorization_provenance": predecessor_authorization,
        },
        "successor": {
            "input_provenance": successor_input,
            "authorization_provenance": successor_authorization,
        },
    }
    return {
        "generator": generator,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "checkpoint": checkpoint,
        "state": state,
        "payload": payload,
        "config_sha256": config_sha256,
        "manifests": manifests,
        "fingerprints": fingerprints,
        "taxonomy": taxonomy,
        "successor_input": successor_input,
        "successor_authorization": successor_authorization,
    }


def _load(case: dict[str, Any], tmp_path: Path, payload: dict[str, Any]) -> TrainingState:
    record = tmp_path / "transition.json"
    record_sha256 = _write_record(record, payload)
    return load_checkpoint_transition(
        case["checkpoint"],
        transition_record_path=record,
        expected_transition_record_sha256=record_sha256,
        model=case["model"],
        optimizer=case["optimizer"],
        scheduler=case["scheduler"],
        scaler=None,
        expected_config_sha256=case["config_sha256"],
        expected_manifest_sha256=case["manifests"],
        expected_manifest_fingerprint=case["fingerprints"],
        expected_taxonomy_sha256=case["taxonomy"],
        expected_input_provenance=case["successor_input"],
        expected_authorization_provenance=case["successor_authorization"],
        expected_run_directory=tmp_path,
        data_generator=case["generator"],
        map_location="cpu",
    )


def test_exact_transition_restores_all_strict_checkpoint_state(tmp_path: Path) -> None:
    case = _case(tmp_path)
    expected_random = torch.rand(3)
    with torch.no_grad():
        case["model"].weight.zero_()
    _ = torch.rand(10)

    state = _load(case, tmp_path, case["payload"])

    assert state == case["state"]
    assert torch.equal(torch.rand(3), expected_random)
    assert not torch.equal(
        case["model"].weight, torch.zeros_like(case["model"].weight)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update({"allowed_change": "ARBITRARY_SOURCE_CHANGE"}),
            "allowed_change",
        ),
        (
            lambda payload: payload["successor"]["input_provenance"].update(
                {"model_revision": "different-model"}
            ),
            "immutable input provenance",
        ),
        (
            lambda payload: payload["successor"]["authorization_provenance"].update(
                {
                    "approval_record_sha256": payload["predecessor"][
                        "authorization_provenance"
                    ]["approval_record_sha256"]
                }
            ),
            "did not refresh approval_record_sha256",
        ),
        (
            lambda payload: payload["training_state"].update({"global_step": 292}),
            "training state",
        ),
    ],
)
def test_transition_rejects_any_unbound_or_non_fast_path_drift(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    case = _case(tmp_path)
    payload = copy.deepcopy(case["payload"])
    mutation(payload)
    with pytest.raises(CheckpointError, match=message):
        _load(case, tmp_path, payload)


def test_transition_rejects_record_hash_mismatch_before_checkpoint_load(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    record = tmp_path / "transition.json"
    _write_record(record, case["payload"])
    with torch.no_grad():
        case["model"].weight.zero_()
    with pytest.raises(CheckpointError, match="transition-record SHA-256"):
        load_checkpoint_transition(
            case["checkpoint"],
            transition_record_path=record,
            expected_transition_record_sha256="9" * 64,
            model=case["model"],
            optimizer=case["optimizer"],
            scheduler=case["scheduler"],
            scaler=None,
            expected_config_sha256=case["config_sha256"],
            expected_manifest_sha256=case["manifests"],
            expected_manifest_fingerprint=case["fingerprints"],
            expected_taxonomy_sha256=case["taxonomy"],
            expected_input_provenance=case["successor_input"],
            expected_authorization_provenance=case["successor_authorization"],
            expected_run_directory=tmp_path,
            data_generator=case["generator"],
            map_location="cpu",
        )
    assert torch.equal(case["model"].weight, torch.zeros_like(case["model"].weight))
