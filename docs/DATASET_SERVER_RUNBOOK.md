# FloodSight real-data server runbook

This performs the Phase 3 **DATA VERIFIED** gate on headless Linux. It does not start
H100 training and assumes neither `sudo` nor a desktop.

## Setup

```bash
cd /data/floodsight
export FLOODSIGHT_DATA_ROOT=/data/floodsight-datasets
export FLOODSIGHT_DATA_CACHE=/data/floodsight-cache

bash scripts/datasets/setup.sh
bash scripts/datasets/doctor.sh
```

## Import official sources

Review each official license/access page and download through its official provider.

```bash
.venv-datasets/bin/python -m floodsight_data.cli import-archive \
  --dataset floodnet --archive /data/imports/floodnet.zip
.venv-datasets/bin/python -m floodsight_data.cli import-archive \
  --dataset rescuenet --archive /data/imports/rescuenet.zip
.venv-datasets/bin/python -m floodsight_data.cli import-directory \
  --dataset visdrone_det --source /data/imports/VisDrone-DET
```

Do not use `--force` until the exact destination is reviewed.

## Inventory and full preparation

```bash
for dataset in floodnet rescuenet visdrone_det; do
  .venv-datasets/bin/python -m floodsight_data.cli --json inspect-source \
    --dataset "$dataset" > "/data/floodsight-datasets/reports/${dataset}-source-inventory.json"
  .venv-datasets/bin/python -m floodsight_data.cli validate --dataset "$dataset"
done

.venv-datasets/bin/python -m floodsight_data.cli taxonomy --write-tables
bash scripts/datasets/prepare.sh --integrity full --materialization hardlink

.venv-datasets/bin/python -m floodsight_data.cli inspect --dataset floodnet --split train --count 24
.venv-datasets/bin/python -m floodsight_data.cli inspect --dataset rescuenet --split train --count 24
.venv-datasets/bin/python -m floodsight_data.cli inspect --dataset visdrone_det --split train --count 24
.venv-datasets/bin/python -m floodsight_data.cli report --all
```

Review every actual label/palette, mapping, ignored label, missing pair, corrupt file,
leak, rare-class sample, random sample, mask alignment, ignored region, small-person
box, and clamped box. Download `reports/` and `inspections/` for human review and
record fingerprints from `manifests/` and `locks/`.

Record review only after that human gate, for each dataset:

```bash
for dataset in floodnet rescuenet visdrone_det; do
  .venv-datasets/bin/python -m floodsight_data.cli review \
    --dataset "$dataset" --reviewer "<reviewer-name>" \
    --license-reviewed --mapping-reviewed --visual-reviewed
done

.venv-datasets/bin/python -m floodsight_data.cli report --all
```

Phase 4 may start only when FloodNet and RescueNet readiness permits segmentation.
Phase 5 has the equivalent VisDrone gate. Until real mapping and visual review are
recorded, the reports remain not ready by design.

```bash
bash scripts/check.sh
git diff --check
git status --short
```

No dataset, archive, output, checkpoint, or run should appear in Git status.
