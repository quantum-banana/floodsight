# FloodSight Parallel Development Handoff State

THIS SNAPSHOT IS FOR PARALLEL APPLICATION/INTEGRATION DEVELOPMENT.
DO NOT USE IT TO ALTER THE ACTIVE VM TRAINING RUN.

- Snapshot timestamp (UTC): `2026-09-01T00:59:00Z`
- Repository source: `/data/floodsight-workspace/floodsight`
- Branch: `phase-3-datasets`
- HEAD: `f272a31ae05e9cb8e532e939e3ca02365755e6a9`
- Dirty working tree: `YES`
- Source fingerprint: `b777a6aa51e5f192bdaf63f15c9ec50c374cb32613296d8fcaf48ac38b945f96`
- Fingerprint scope: 281 included working-tree files before adding this handoff note
- Fingerprint algorithm: SHA-256 over the bytewise path-sorted, NUL-delimited tuples `type, relative path, POSIX mode, byte size, file SHA-256`.

## Git status

```text
## phase-3-datasets...origin/phase-3-datasets [ahead 1]
 M .gitignore
?? ml/detection/README.md
?? ml/detection/configs/
?? ml/detection/floodsight_detection/
?? ml/detection/tests/
?? ml/segmentation/README.md
?? ml/segmentation/configs/
?? ml/segmentation/floodsight_segmentation/
?? ml/segmentation/tests/
?? ml/training/
?? scripts/training/
?? shared/taxonomy/floodnet-mapping-v2.yaml
?? shared/taxonomy/rescuenet-mapping-v2.yaml
?? shared/taxonomy/segmentation-taxonomy-v2.yaml
```

## Git diff --stat

```text
 .gitignore | 1 +
 1 file changed, 1 insertion(+)
```

## Modified and untracked repository files

```text
 M .gitignore
?? ml/detection/README.md
?? ml/detection/configs/yolo11l_h100_max_quality.yaml
?? ml/detection/floodsight_detection/__init__.py
?? ml/detection/floodsight_detection/__main__.py
?? ml/detection/floodsight_detection/approval.py
?? ml/detection/floodsight_detection/checkpointing.py
?? ml/detection/floodsight_detection/cli.py
?? ml/detection/floodsight_detection/config.py
?? ml/detection/floodsight_detection/contract.py
?? ml/detection/floodsight_detection/determinism.py
?? ml/detection/floodsight_detection/errors.py
?? ml/detection/floodsight_detection/hashing.py
?? ml/detection/floodsight_detection/real_smoke.py
?? ml/detection/floodsight_detection/real_smoke_worker.py
?? ml/detection/floodsight_detection/runs.py
?? ml/detection/floodsight_detection/runtime.py
?? ml/detection/floodsight_detection/smoke.py
?? ml/detection/floodsight_detection/ultralytics_runtime.py
?? ml/detection/floodsight_detection/weights.py
?? ml/detection/tests/conftest.py
?? ml/detection/tests/test_config_runs_cli.py
?? ml/detection/tests/test_contract.py
?? ml/detection/tests/test_security_and_real_smoke.py
?? ml/detection/tests/test_smoke.py
?? ml/segmentation/README.md
?? ml/segmentation/configs/segformer_b2.yaml
?? ml/segmentation/floodsight_segmentation/__init__.py
?? ml/segmentation/floodsight_segmentation/__main__.py
?? ml/segmentation/floodsight_segmentation/approval.py
?? ml/segmentation/floodsight_segmentation/artifact.py
?? ml/segmentation/floodsight_segmentation/checkpoint.py
?? ml/segmentation/floodsight_segmentation/checkpoint_probe.py
?? ml/segmentation/floodsight_segmentation/cli.py
?? ml/segmentation/floodsight_segmentation/config.py
?? ml/segmentation/floodsight_segmentation/dataset.py
?? ml/segmentation/floodsight_segmentation/engine.py
?? ml/segmentation/floodsight_segmentation/errors.py
?? ml/segmentation/floodsight_segmentation/guard.py
?? ml/segmentation/floodsight_segmentation/integrity.py
?? ml/segmentation/floodsight_segmentation/manifest.py
?? ml/segmentation/floodsight_segmentation/metrics.py
?? ml/segmentation/floodsight_segmentation/model.py
?? ml/segmentation/floodsight_segmentation/optim.py
?? ml/segmentation/floodsight_segmentation/prepared.py
?? ml/segmentation/floodsight_segmentation/reproducibility.py
?? ml/segmentation/floodsight_segmentation/runtime.py
?? ml/segmentation/floodsight_segmentation/smoke.py
?? ml/segmentation/floodsight_segmentation/supervision.py
?? ml/segmentation/floodsight_segmentation/threaded_loader.py
?? ml/segmentation/floodsight_segmentation/transforms.py
?? ml/segmentation/floodsight_segmentation/transition.py
?? ml/segmentation/tests/conftest.py
?? ml/segmentation/tests/test_checkpoint_transition.py
?? ml/segmentation/tests/test_config_manifest_cli.py
?? ml/segmentation/tests/test_ml_contracts.py
?? ml/segmentation/tests/test_prepared_fastpath.py
?? ml/training/README.md
?? ml/training/accepted-wheelhouse.sha256
?? ml/training/audit-canonical-requirements.txt
?? ml/training/requirements-py312-cu130.lock
?? ml/training/requirements-pypi.txt
?? ml/training/requirements-torch-cu130.txt
?? ml/training/ultralytics-runtime-assets-v1.json
?? ml/training/ultralytics-runtime-assets-v1.sha256
?? ml/training/ultralytics-settings-v1.json
?? ml/training/verify-locked-environment.py
?? scripts/training/check.sh
?? scripts/training/run-locked.sh
?? scripts/training/runtime-offline.sh
?? scripts/training/setup-locked.sh
?? scripts/training/setup.sh
?? shared/taxonomy/floodnet-mapping-v2.yaml
?? shared/taxonomy/rescuenet-mapping-v2.yaml
?? shared/taxonomy/segmentation-taxonomy-v2.yaml
```

## Snapshot exclusions

Excluded from this code-only handoff: `.git`, all local Python environments, `node_modules`, generated `dist`/`build` output, Python/test/lint caches, real `.env` and credential files, dataset payloads and caches, raw/processed data, training runs, manifests/reports/logs, checkpoints, model weights, downloaded wheelhouses, archives, databases, and temporary/runtime files. Safe `.env.example` templates and empty-directory `.gitkeep` placeholders are retained.
