# FloodSight detection training infrastructure

This subtree is the isolated Phase 5 YOLO training boundary. It consumes only a
completed corrected Phase 3 `visdrone_det-detection_v2` manifest. It never imports
`floodsight_data`, and importing this package does not import PyTorch or
Ultralytics.

The accepted runtime is exact: Python 3.12.3, Ultralytics `8.3.222`, Torch
`2.13.0+cu130`, Torchvision `0.28.0+cu130`, and the complete 103-distribution
resolved lock. The launcher and Python runtime both verify the canonical venv,
acceptance-marker SHA-256, resolved-lock SHA-256, and full installed snapshot.
A nearby version is a blocking error, not an implicit compatibility claim.

## Frozen contract

The contract gate requires:

- the exact canonical VisDrone v2 manifest path, SHA-256, dataset fingerprint,
  source version, preparation version, and taxonomy/mapping identities;
- `full` manifest integrity, by default;
- unique samples, paths, and image content with unchanged SHA-256 values;
- preserved source/target splits, including non-empty `train` and `val`;
- IDs `0..7` with the exact names in `detection-taxonomy-v1`;
- every configured class represented in the training split;
- every target label hash, row, class ID, finite normalized box, manifest class
  count, and per-object provenance record to agree.

Freezing creates a new, collision-protected run-local YOLO view. Validated source
images and YOLO label files are materialized as independent read-only copies
(using copy-on-write reflinks when available). This also makes Phase 3 `manifest-only` image
materialization safe for Ultralytics, whose loader expects parallel `images/`
and `labels/` paths. The source dataset is never modified.

The primary single-H100 configuration is
`configs/yolo11l_h100_max_quality.yaml`. It uses pretrained `yolo11l.pt` at
1536 px and preserves the eight detection-training subclasses; product-level
person/vehicle aggregation remains a later inference concern.

The host exposes only 64 MiB at `/dev/shm`, while one 12-image 1536 px RGB
batch is already 81 MiB as uint8 before labels or prefetch. The frozen loader
therefore uses `workers: 0`, keeping loading/augmentation in the training
process and avoiding PyTorch worker IPC bus errors. This changes throughput,
not the selected samples or augmentation policy.

That same config freezes the full-training root to
`/data/floodsight-workspace/floodsight-datasets/runs/detection` and the bounded
real-smoke root to
`/data/floodsight-workspace/floodsight-datasets/runs/detection-real-smoke`.
Every run or real smoke must be one new direct child of its respective root.
Full resume accepts only the approved direct child's exact `weights/last.pt`,
and only when its bytes match the latest immutable epoch generation, hashed
sidecar, crash-safe pointer, run identity, frozen trainer arguments, dataset
YAML, and saved RNG state;
the CLI cannot override either root, and a nonblocking process-lifetime lock
prevents concurrent train/resume writers for one run.

The configuration records only the required filename. Production code never
passes that basename to Ultralytics. A separate weight-audit JSON must provide
an explicit absolute local path, SHA-256, official source/release provenance,
and completed license review. The hash is re-read before any model load, so
there is no implicit download or network fallback.

## Import-safe audit commands

From the repository root in the pinned ML environment:

```bash
export PYTHONPATH="$PWD/ml/detection"

bash scripts/training/run-locked.sh detection validate \
  --config ml/detection/configs/yolo11l_h100_max_quality.yaml \
  --manifest "$FLOODSIGHT_DATA_ROOT/manifests/visdrone_det-detection_v2.json" \
  --data-root "$FLOODSIGHT_DATA_ROOT"

bash scripts/training/run-locked.sh detection freeze \
  --config ml/detection/configs/yolo11l_h100_max_quality.yaml \
  --manifest "$FLOODSIGHT_DATA_ROOT/manifests/visdrone_det-detection_v2.json" \
  --data-root "$FLOODSIGHT_DATA_ROOT" \
  --output /explicit/new/audit-directory/detection-contract
```

The synthetic smoke is opt-in, writes only beneath a brand-new output path, uses
generated 64x64 images and `yolo11n.yaml`, and never downloads weights:

```bash
bash scripts/training/run-locked.sh detection smoke \
  --output /explicit/new/smoke-directory \
  --device cuda:0 \
  --allow-synthetic-smoke
```

It must prove loader, model forward, loss, backward, validation, checkpoint, and
resume. A synthetic PASS means code-ready only, never data-verified or
model-ready.

After the real manifest and local weight audit are frozen, a separately gated
smoke selects at most one deterministic training sample per missing class plus
at most two validation samples. The frozen configured batch remains 12, while
Ultralytics explicitly clamps the actual loader batch to
`min(12, bounded_subset_size)`; both values and that policy are attested. The
complete training subset is at most eight samples, so the initial epoch has
exactly one batch. The parent always stops after that batch. A fresh process may
then execute at most seven additional batches and at most seven optimizer calls,
stopping immediately at the first applied update. Total train batches and
optimizer calls are independently guarded before execution at **eight each**;
the evidence allows fewer optimizer calls than batches if gradient accumulation
requires it. The larger bound follows an empirical four-call AMP exhaustion at
scale 4096 and remains a hard ceiling, not a target. Every preceding optimizer
call must prove a scaler overflow backoff and no underlying AdamW step; the final
call must prove exactly one applied AdamW update. If the parent unexpectedly
applies that update, the gate fails closed before spawning the resume process,
so it cannot apply a second update merely to prove resume. Thus the smoke applies
exactly one parameter update or fails, never running the configured 200-epoch
training job:

```bash
bash scripts/training/run-locked.sh detection real-smoke \
  --config ml/detection/configs/yolo11l_h100_max_quality.yaml \
  --manifest "$FLOODSIGHT_DATA_ROOT/manifests/visdrone_det-detection_v2.json" \
  --data-root "$FLOODSIGHT_DATA_ROOT" \
  --weights-audit /absolute/audited/yolo11l-weight-audit.json \
  --output /data/floodsight-workspace/floodsight-datasets/runs/detection-real-smoke/<new-name> \
  --device 0 \
  --allow-real-smoke
```

This path requires the accepted offline H100 runtime and a `PYTHONHASHSEED`
that was fixed before interpreter launch. It loads the audited local
`yolo11l.pt` and preserves every frozen train/validation setting, including
`epochs: 200`; there are no argument overrides. External callbacks enforce the
batch and optimizer-call ceilings. The report records the unchanged
effective-argument hash, exact executable transform configuration (including
Mosaic, MixUp, CopyPaste, CutMix, flip, HSV, perspective, and BGR settings),
checkpoint SHA-256, restored optimizer param groups and AMP scaler, and each
optimizer call's epoch, scheduler position, accumulation factor, configured and
loader batch sizes, and learning rates. It also records train-batch,
optimizer-call, backward-hook, and validation-run counts so repeated validation
after AMP backoffs remains explicit. Every non-`None` gradient received by the
backward hook counts as one executed backward call, independently of whether
that gradient is finite. A separate `backward_gradient_finiteness` trace records
one boolean per training batch: nonfinal entries may be false during expected
AMP overflow backoff, while the batch producing the sole applied update must be
the final true entry. The top-level v5 trace exactly mirrors its backend trace,
and the signed v3 worker trace is cross-bound to the resume-phase suffix of that
backend trace.

The resume restores Python, NumPy, Torch, CUDA, and loader RNG state from the
hashed worker request in a new interpreter. It must prove a distinct PID and
that its parent PID is the initial smoke PID. Python object IDs are retained only
as positive process-local diagnostics and may be numerically equal across the
two processes. The process-result file and resumable checkpoint are rehashed
both when the smoke report is created and when it is later accepted for
training. Before the child starts, the parent must prove that its CUDA training
allocation was released, preventing two training footprints from overlapping
on the shared H100.

This pre-review smoke accepts weight provenance marked either
`PENDING_HUMAN_SIGNOFF` or `APPROVED_FOR_RESEARCH_DEMO` and records that state in
its report. The standard completed-review training path accepts only
`APPROVED_FOR_RESEARCH_DEMO`; the separately validated user-deferral path is
documented below. Both require a distinct, hash-bound full-training approval.

## Real training lock

The `train` command fails before reading configuration/data or importing ML
frameworks unless `--allow-training` is explicitly present. That boolean alone
is insufficient: `--weights-audit`, `--real-smoke-report`, and `--approval` are
also mandatory. The
approval JSON must say `APPROVE_FULL_TRAINING`, include all responsible-use and
review acknowledgements, and match the exact run name, config SHA-256, manifest
SHA-256, dataset fingerprint, executable training-source SHA-256, and
pretrained-weight SHA-256, exact passing real-smoke report path and SHA-256,
plus the exact reviewed report path and SHA-256. The runtime checks
authorization again, requires exact dependency versions, reserves a new run
directory atomically, and accepts resume only from that run's exact
`weights/last.pt` with unchanged frozen identities. Each save publishes a
read-only content-addressed checkpoint generation before updating the durable
pointer. Resume rejects a symlink, foreign/plain checkpoint, torn live file,
sidecar drift, unsafe saved-argument redirection, or RNG-state drift.

When the user explicitly defers rather than completes visual/provenance review,
the truthful alternative is approval schema
`floodsight-full-training-approval-v5`. It binds an exact
`floodsight-user-training-launch-override-v1` JSON record and requires the
deferred-review acknowledgement set; it must not claim dataset, label, or
license review completion. A pending pretrained-weight license status is
accepted only after that exact override is independently revalidated at both
the CLI and execution boundaries. All technical, identity, real-smoke,
responsible-use, output-containment, and resume checks remain unchanged.
`--output-root` must resolve
exactly to `/data/floodsight-workspace/floodsight-datasets/runs/detection`; the
run name selects its one exclusive direct child.

Do not use `--allow-training` during infrastructure, dataset-audit, synthetic
smoke, or human-review stages. Full training is a separate final authorization.

## Verification

```bash
PYTHONPATH=ml/detection python -m pytest -q ml/detection/tests
ruff check ml/detection
```

The pure tests run without PyTorch or Ultralytics. The explicit synthetic smoke
is the separate integration test for the pinned ML environment.
