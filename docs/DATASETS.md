# FloodSight datasets

Phase 3 prepares three external public datasets without redistributing or training
on them.

| Dataset | Role | Official source | License review |
| --- | --- | --- | --- |
| FloodNet | Post-flood semantic segmentation | [BinaLab/FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | User acceptance and deployment review required |
| RescueNet | Damage, road-state, water, and scene segmentation | [BinaLab/RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | `REVIEW_REQUIRED`; official terms state CC BY-NC-ND |
| VisDrone-DET | Aerial people and vehicle boxes | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | Review required before use |

FloodNet and RescueNet map into the segmentation contract. VisDrone-DET remains a
separate detection contract. They are never forced into one false universal format.

## External layout

```text
<FLOODSIGHT_DATA_ROOT>/
├── raw/                 immutable imported source data
│   ├── floodnet/
│   ├── rescuenet/
│   └── visdrone_det/
├── interim/             normalized source structures, source labels preserved
├── processed/
│   ├── segmentation_v1/
│   └── detection_v1/
├── manifests/           deterministic JSON and JSONL inventories
├── reports/             machine and human-readable health results
├── inspections/         deterministic contact sheets and indexes
└── locks/               external acquisition/preparation metadata
```

`raw/` is never modified by conversion. Generated content stays outside Git.

## Acquisition and import

Official sources require license/access review or provider-hosted manual steps.
`acquire` reports the follow-up; it does not scrape around gates.

```powershell
$env:FLOODSIGHT_DATA_ROOT = "D:\FloodSight-Datasets"
$env:FLOODSIGHT_DATA_CACHE = "D:\FloodSight-Cache"
.\.venv-datasets\Scripts\python.exe -m floodsight_data.cli acquire --dataset rescuenet
.\.venv-datasets\Scripts\python.exe -m floodsight_data.cli import-archive `
  --dataset rescuenet --archive D:\Downloads\rescuenet.zip
```

```bash
export FLOODSIGHT_DATA_ROOT=/data/floodsight-datasets
export FLOODSIGHT_DATA_CACHE=/data/floodsight-cache
.venv-datasets/bin/python -m floodsight_data.cli import-directory \
  --dataset visdrone_det --source /data/imports/VisDrone-DET
```

Imports validate archive type and members, stage before replacement, reject path
traversal and links, and do not overwrite raw data unless `--force` is explicit.
Interrupted stable-URL downloads retain `.part` files where resume is supported.

## Inspection, validation, and conversion

```text
python -m floodsight_data.cli inspect-source --dataset floodnet
python -m floodsight_data.cli validate --dataset floodnet
python -m floodsight_data.cli prepare --dataset floodnet --integrity full
```

Inspection reports source labels, mask representation, extensions, splits,
dimensions, pairs, corruption, empty annotations, and distributions. Unknown IDs,
colours, or VisDrone classes are blocking and never become background silently.

Segmentation conversion preserves dimensions and official splits, keeps original
image bytes, writes indexed PNG, and encodes ignored pixels as 255. VisDrone
conversion validates all eight fields, normalizes boxes, reports clamping, preserves
truncation/occlusion metadata, and writes valid empty label files.

Manifests retain dataset provenance, relative paths, stable IDs/order, dimensions,
hashes, class counts, versions, and a reproducible fingerprint. Full validation
reports duplicates, renamed duplicates, conflicting masks, and cross-split leakage
without deleting or moving sources.

Readiness requires successful import, complete pairs/conversions, required splits,
no blocking errors, reviewed mappings, and human-reviewed visual samples. A local
`MISSING` status is expected when public datasets have not been imported.

After reviewing the exact license terms, mappings, and generated inspection images,
record the attestation against the current fingerprint. A later conversion changes the
fingerprint and invalidates stale review metadata automatically:

```text
python -m floodsight_data.cli review --dataset floodnet --reviewer <name> \
  --license-reviewed --mapping-reviewed --visual-reviewed
```

## Troubleshooting

- `data_root_missing`: set `FLOODSIGHT_DATA_ROOT` or pass `--data-root`.
- `destination_exists`: review existing raw data; use `--force` only intentionally.
- `dataset_structure_unrecognized`: compare the archive with the registry.
- `unknown_mask_ids` / `unknown_mask_colors`: review the actual palette.
- `pairing_failed`: repair missing or conflicting source pairs.
- `invalid_bounding_box`: inspect the exact annotation line.
- `prepare_all_incomplete`: one or more real datasets are absent or blocked.
