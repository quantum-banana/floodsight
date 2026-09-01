# FloodSight ML training environment

This directory defines the isolated Phase 4/5 environment used by the SegFormer
and YOLO training packages. It deliberately does not add ML frameworks to the
lightweight Phase-3 `floodsight-data` package.

PyTorch and torchvision are isolated in `requirements-torch-cu130.txt` and are
installed only from the explicit official PyTorch CUDA index. All remaining direct
dependencies are pinned in `requirements-pypi.txt` and installed only from PyPI;
this avoids cross-index dependency confusion. The complete transitive resolution
is frozen in `requirements-py312-cu130.lock`, and setup refuses an environment
whose installed snapshot differs from it.

Setup source-routes Torch, torchvision, Triton, and their CUDA runtime wheels to
the official PyTorch index; every other resolved wheel is fetched from PyPI.
Both installations use `--no-deps`, so dependency resolution cannot silently
cross either source boundary. `pip check` then proves the assembled graph is
complete.
The baseline targets Python 3.12, NVIDIA H100
GPUs, and PyTorch CUDA 13.0 wheels. Production commands remain training-locked by default. Building the
environment or running unit/synthetic smoke tests must not access real datasets
or launch real training.

Create or reconcile a source-routed development environment:

```bash
bash scripts/training/setup.sh
```

The immutable accepted environment is instead built from the external,
SHA-256-inventoried wheelhouse. This command refuses an existing target and
writes its acceptance marker only after offline installation, exact-lock,
dependency, vulnerability, CUDA, and H100 checks all pass:

```bash
bash scripts/training/setup-locked.sh
```

Run the source and synthetic checks after both training packages are present:

```bash
bash scripts/training/check.sh
```

Invoke either stack through the locked launcher. It verifies the accepted
environment marker, enforces offline runtime settings, and removes the host's
unrelated cuDNN path before importing PyTorch:

```bash
bash scripts/training/run-locked.sh segmentation --help
bash scripts/training/run-locked.sh detection --help
```

The launcher also verifies the SHA-256-pinned Ultralytics font and `yolo11n.pt`
AMP-check assets before import, installs the font into an isolated settings
directory, and points Ultralytics' auxiliary weights lookup at the audited
cache. This removes the package's otherwise implicit font and AMP-check
downloads; `YOLO_OFFLINE=true` remains enabled as a second boundary.

Actual training requires the code-level training guard, frozen manifests,
completed real-data smoke tests, and recorded human approval. The setup and
check scripts do not provide that approval.
