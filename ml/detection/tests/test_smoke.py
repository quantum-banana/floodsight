from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.smoke import run_synthetic_smoke


class PassingBackend:
    def run(
        self,
        *,
        data_yaml: Path,
        output_root: Path,
        seed: int,
        device: str,
    ) -> dict[str, Any]:
        assert data_yaml.is_file()
        assert seed == 17
        assert device == "cpu"
        output_root.mkdir(parents=True)
        checkpoint = output_root / "weights/last.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic checkpoint")
        return {
            "loader": True,
            "model_forward": True,
            "loss": True,
            "backward": True,
            "validation": True,
            "checkpoint": True,
            "resume": True,
            "initial_checkpoint": str(checkpoint),
            "resumed_checkpoint": str(checkpoint),
        }


class IncompleteBackend(PassingBackend):
    def run(self, **kwargs: Any) -> dict[str, Any]:
        result = super().run(**kwargs)
        result["backward"] = False
        return result


def test_smoke_orchestration_is_generated_data_only(tmp_path: Path) -> None:
    report = run_synthetic_smoke(
        tmp_path / "smoke",
        allow_synthetic_smoke=True,
        seed=17,
        backend=PassingBackend(),
    )

    assert report["status"] == "PASS"
    assert report["synthetic_only"] is True
    assert report["real_dataset_accessed"] is False
    assert report["real_training_started"] is False
    assert all(report["checks"].values())
    assert Path(report["report_path"]).is_file()


def test_smoke_fails_closed_when_backend_does_not_prove_every_step(tmp_path: Path) -> None:
    with pytest.raises(DetectionInfrastructureError) as error:
        run_synthetic_smoke(
            tmp_path / "smoke",
            allow_synthetic_smoke=True,
            seed=17,
            backend=IncompleteBackend(),
        )

    assert error.value.code == "synthetic_smoke_incomplete"


def test_smoke_refuses_output_collision(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    output.mkdir()

    with pytest.raises(DetectionInfrastructureError) as error:
        run_synthetic_smoke(
            output,
            allow_synthetic_smoke=True,
            backend=PassingBackend(),
        )

    assert error.value.code == "smoke_collision"
