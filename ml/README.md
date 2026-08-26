# FloodSight dataset tooling

This lightweight, training-framework-free Phase 3 package handles FloodNet,
RescueNet, and VisDrone-DET acquisition support, source inspection, validation,
conversion, deterministic manifests, fingerprints, reports, and visual inspection.

It deliberately excludes PyTorch, SegFormer, YOLO, training, inference, and
checkpoints. Importing `floodsight_data` neither requires nor creates a data root.

## Install

Python 3.11 or newer is required. The scripts create a separate `.venv-datasets`:

```powershell
.\scripts\datasets\setup.ps1
$env:FLOODSIGHT_DATA_ROOT = "D:\FloodSight-Datasets"
$env:FLOODSIGHT_DATA_CACHE = "D:\FloodSight-Cache"
.\scripts\datasets\doctor.ps1
```

```bash
bash scripts/datasets/setup.sh
export FLOODSIGHT_DATA_ROOT=/data/floodsight-datasets
export FLOODSIGHT_DATA_CACHE=/data/floodsight-cache
bash scripts/datasets/doctor.sh
```

## CLI

```text
python -m floodsight_data.cli doctor
python -m floodsight_data.cli registry
python -m floodsight_data.cli taxonomy
python -m floodsight_data.cli acquire --dataset floodnet --data-root <path>
python -m floodsight_data.cli import-archive --dataset rescuenet --archive <path> --data-root <path>
python -m floodsight_data.cli import-directory --dataset visdrone_det --source <path> --data-root <path>
python -m floodsight_data.cli inspect-source --dataset floodnet --data-root <path>
python -m floodsight_data.cli validate --dataset floodnet --data-root <path>
python -m floodsight_data.cli convert --dataset floodnet --data-root <path> --integrity full
python -m floodsight_data.cli manifest --dataset floodnet --data-root <path>
python -m floodsight_data.cli inspect --dataset floodnet --split train --count 24 --data-root <path>
python -m floodsight_data.cli review --dataset floodnet --reviewer <name> --license-reviewed --mapping-reviewed --visual-reviewed --data-root <path>
python -m floodsight_data.cli report --all --data-root <path>
python -m floodsight_data.cli fingerprint --dataset floodnet --data-root <path>
python -m floodsight_data.cli prepare --dataset floodnet --data-root <path>
python -m floodsight_data.cli prepare-all --data-root <path> --integrity full
```

Put global `--json` or `--debug` before the subcommand. Expected user errors are
structured and do not print tracebacks. `prepare-all` never downloads data.

## Design

- Typed registry records load from `configs/datasets/`.
- External paths are resolved only for requested commands.
- Imports reject traversal, links, unsafe archive entries, and silent replacement.
- Source adapters inventory exact labels before conversion.
- Declarative files in `shared/taxonomy/` are the mapping source of truth.
- Segmentation conversion writes atomic indexed PNG masks with ignore index 255.
- VisDrone conversion retains eight target subclasses and source metadata while
  emitting normalized YOLO labels.
- Manifests use relative paths, stable IDs/order, schemas, and fingerprints.
- Reports derive readiness from checks; they never hardcode a ready result.
- Inspection selection and contact sheets are deterministic.

`manifest-only`, `hardlink`, and `copy` materialization are supported. Hardlink is
the default and falls back to copy when the filesystem cannot link.

Integrity `fast` records path, size, modification metadata, and annotation hashes.
Integrity `full` hashes every relevant image and annotation and is required for the
real-data leakage gate.

## Test

```powershell
.\.venv-datasets\Scripts\python.exe -m ruff check ml\floodsight_data ml\tests
.\.venv-datasets\Scripts\python.exe -m pytest ml\tests
```

Tests generate tiny images, masks, annotations, archives, duplicates, and leaks.
No public dataset or large binary fixture is required.
