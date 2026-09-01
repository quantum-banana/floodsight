# FloodSight SegFormer training infrastructure

This subtree is the isolated Phase-4 semantic-segmentation stack. It consumes
only SHA-256-locked Phase-3 manifests and external files under an explicit data
root. It does not import or modify `floodsight_data`, and no dataset, checkpoint,
run, cache, or generated report belongs in Git.

The frozen baseline is
`nvidia/segformer-b2-finetuned-ade-512-512` at upstream revision
`de01bae28967510f9ddd496c60a969357195400c`. The upstream
`pytorch_model.bin` is bound to SHA-256
`187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3`.
Production loading accepts only the exact audited local conversion named
`model.safetensors` (SHA-256
`4a7ab8f05afe62dfdd75338b7fc2eb10ad1347bf5ade78ca2109951f5c717b86`)
and its exact adjacent provenance record (SHA-256
`197a2a29f580406fc7d606445ffcba93bcf5d76dde3d4be6e2391c23a0a27add`).
Their canonical absolute paths and hashes are frozen in the configuration; CLI
arguments must match them exactly. Loading is `local_files_only`, uses
`trust_remote_code=False`, and has no network or `.bin` fallback.

The B2 input crop is 1024×1024. Training samples a 50–100% source-area crop
with a shared image/mask geometry before resizing; semantic masks always use
nearest-neighbor interpolation. The single-H100 baseline uses batch size 4 and
four-step accumulation for effective batch size 16. It fine-tunes in the unified
16-class FloodSight taxonomy. Mask ID `255` is ignored. Semantically different
source labels remain distinct: non-flooded is not equated with clear/no-damage,
tree is not broadened to grass, and Pool is never collapsed into Water. Partial supervision is
mandatory:

- FloodNet supports IDs `0,1,2,3,6,7,12,13,14,15`.
- official RescueNet supports IDs `0,1,4,5,8,9,10,11,12,13,15`.

For every sample, unsupported logits are removed from the softmax denominator.
Any non-ignore mask ID outside that sample's source-dataset support is a hard
error. This prevents missing labels from being trained as negatives or silently
mapped to background.

The accepted host exposes only 64 MiB of `/dev/shm`, while one collated batch is
about 80 MiB before multiprocessing overhead. Both full loaders therefore freeze
`num_workers: 0`; in-process loading preserves data and model quality while
trading some throughput for protection against worker IPC bus errors.

The frozen configuration also binds the exact absolute paths and SHA-256 values
of `segmentation-taxonomy-v2.yaml`, `floodnet-mapping-v2.yaml`, and
`rescuenet-mapping-v2.yaml`. Manifests use fingerprint algorithm
`sha256-canonical-manifest-identity-v1`: SHA-256 over UTF-8 canonical JSON
(`sort_keys=True`, separators `(',', ':')`) containing the immutable root fields
and samples sorted by `sample_id`; `created_at` and `fingerprint` are excluded.
The stored value must equal this recomputation and the caller-supplied expected
fingerprint.

Loss is frozen as dataset-masked weighted cross-entropy with an explicit
16-value positive class-weight vector and normalization by summed valid-pixel
weights. The vector is the Stage-10 retained-training inverse-square-root
proposal, floored at 0.25 and capped at 4.0, in target-ID order 0 through 15.
The configuration binds that vector to the exact Stage-10 source report at
`/data/floodsight-workspace/floodsight-datasets/reports/pretraining_gate/segmentation_stages09_10_20260831T131322Z_v1/stage10/stage10_post_sanitation_class_balance_and_weights.json`
(SHA-256
`1cfcdacdb1f08254170c3aeba458cb4aa85cd3d663bd4d220ef9fb8354733869`)
and refuses if the report bytes or its target-ID-ordered proposal drift.
Sampling is explicitly replacement-based, dataset-balanced 50/50, with one
sample draw per retained training-manifest row.

Final Stage-9 manifests use preparation
`segmentation_v2_20260831T131322Z_v1`. The reader requires the exact final row
contract, including source schema, mapping version/hash, sorted supervision
classes, ignore semantics, and included disposition. Dataset loading hashes and
decodes the source indexed mask, applies the audited source-to-target ID map,
and requires pixel-exact agreement with the target mask and declared counts.

## Safety boundary

CLI import and help do not require Torch. The `train` and `validate` commands
refuse before opening any manifest, checkpoint, or dataset unless the operator
passes `--allow-training`. A boolean is deliberately insufficient: a separate
content-addressed JSON approval record must say `approval_kind: HUMAN`, say
`decision: APPROVED`, authorize the exact operation, and bind all of these:

- frozen configuration SHA-256;
- executable training-source SHA-256;
- every absolute manifest path, SHA-256, and recomputed fingerprint;
- exact taxonomy/mapping paths and SHA-256 values;
- the exact approved run directory;
- model ID, exact upstream revision, local safetensors path/SHA-256, and
  conversion-provenance path/SHA-256;
- a real human-review report path and SHA-256.

The review report is itself strict JSON with schema
`floodsight-segmentation-human-review-v1`, an `APPROVED` decision, no open
blockers, and explicit acknowledgements for taxonomy/Pool semantics, partial
supervision, leakage, real smoke, provenance/license discrepancy, and the
no-training boundary.

An explicit user launch override may instead truthfully defer those review
items without marking them complete. That path uses the exact hash-bound
`floodsight-user-training-launch-override-v1` record, requires
`human_review_status` and `provenance_review_status` to be
`DEFERRED_BY_USER`, requires both corresponding `*_completed` fields to remain
false, and must explicitly authorize both production models, persistent tmux,
and the final source freeze. The outer training approval remains a separate
`HUMAN` `APPROVED` record bound to the run; the deferral changes no technical
artifact, smoke, manifest, model, or configuration requirement.

The model conversion record must have schema
`floodsight-model-artifact-v1`, source revision/source `.bin` hash, local
safetensors hash, conversion timestamp/tool, and `audit_status: PASS`. The
pre-review artifact record also carries
`human_review_status: PENDING_HUMAN_SIGNOFF`. The later approval record must have
schema `floodsight-training-approval-v3`. It also binds the exact passing
bounded-real-smoke report path and SHA-256; that report is revalidated against
the current source, canonical train manifests, taxonomy, local model,
single-step envelope, checkpoint hash, and fresh-process resume proof. Checkpoints
and run history retain the model, approval, review, config, and manifest hashes,
and exact resume rejects any drift.

Full-training outputs are restricted to one direct child of
`/data/floodsight-workspace/floodsight-datasets/runs/segmentation`. A resume may
only use that same approved run's exact `last.pt`; `best.pt`, copied checkpoints,
and checkpoints from another directory are rejected.
Bounded real-smoke outputs are separately restricted to one direct child of
`/data/floodsight-workspace/floodsight-datasets/runs/segmentation-real-smoke`.
Every real run holds a non-blocking process-lifetime lock on its output
directory so a duplicate resume cannot overwrite history or checkpoints.

The only currently authorized executable path is the generated-tensor smoke:

```bash
bash scripts/training/run-locked.sh segmentation smoke --device cpu
```

It constructs a tiny random SegFormer configuration offline, performs one
synthetic optimization step, calculates metrics, and verifies checkpoint/RNG
resume. Its output and checkpoint are labelled `DEMO_SIMULATED`; it does not
open a real manifest or dataset and is not model training.

After manifests and converted weights pass their technical gates, the separate
bounded real-data smoke is available before human sign-off:

```text
bash scripts/training/run-locked.sh segmentation real-smoke \
  --data-root <external-root> \
  --manifest <floodnet-manifest.json> --manifest-sha256 <sha256> \
  --manifest-fingerprint <fingerprint> \
  --manifest <rescuenet-manifest.json> --manifest-sha256 <sha256> \
  --manifest-fingerprint <fingerprint> \
  --model-safetensors <audited-dir/model.safetensors> \
  --model-safetensors-sha256 <sha256> \
  --model-provenance-record <audited-dir/provenance.json> \
  --model-provenance-record-sha256 <sha256> \
  --output-dir <external-smoke-directory> \
  --device cuda:0 \
  --allow-real-smoke
```

That command is H100/BF16-only and deterministically selects at most two
training samples per source dataset to cover Pool and a source-specific class.
It prefers one sample that jointly covers both requirements, verifies the
actual post-training-transform masks still contain those semantics, and
requires positive finite loss/gradients and an actual parameter change,
performs exactly one optimizer step, and validates on the same bounded samples
as a loader/metric check (never a benchmark). Its production checkpoint is
reconstructed and loaded in a fresh Python process, which compares exact model,
optimizer, scheduler, scaler, Python/NumPy/Torch CPU+CUDA/data-generator RNG,
training state, and a child-side recomputation of the executable-source
fingerprint. It has no epoch loop and cannot launch full training.
Its report is explicitly `PENDING_HUMAN_SIGNOFF` and `training_authorized: false`;
reviewers use that evidence to decide whether to issue the later training
approval record.

Future gated training requires paired manifest paths and hashes, for example:

```text
bash scripts/training/run-locked.sh segmentation train \
  --data-root <external-root> \
  --train-manifest <floodnet-manifest.json> \
  --train-manifest-sha256 <sha256> \
  --train-manifest-fingerprint <fingerprint> \
  --train-manifest <rescuenet-manifest.json> \
  --train-manifest-sha256 <sha256> \
  --train-manifest-fingerprint <fingerprint> \
  --validation-manifest <floodnet-manifest.json> \
  --validation-manifest-sha256 <sha256> \
  --validation-manifest-fingerprint <fingerprint> \
  --validation-manifest <rescuenet-manifest.json> \
  --validation-manifest-sha256 <sha256> \
  --validation-manifest-fingerprint <fingerprint> \
  --model-safetensors <audited-dir/model.safetensors> \
  --model-safetensors-sha256 <sha256> \
  --model-provenance-record <audited-dir/provenance.json> \
  --model-provenance-record-sha256 <sha256> \
  --approval-record <approval.json> \
  --approval-record-sha256 <sha256> \
  --output-dir <external-run-directory> \
  --device cuda:0 \
  --allow-training
```

The approval must separately authorize `TRAIN` (and `VALIDATE` for standalone
validation). `REAL_SMOKE` approval never authorizes an epoch-based run.

## Tests

Pure configuration, manifest, path-safety, and guard tests run without ML
dependencies. Torch/Transformers tests generate all images, masks, and tensors
inside temporary directories and are skipped when the isolated ML environment
is unavailable.

```bash
PYTHONPATH=ml/segmentation python -m pytest ml/segmentation/tests
```
