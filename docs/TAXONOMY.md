# FloodSight taxonomy candidate

Phase 3 keeps product, segmentation-training, and detection-training taxonomies
distinct. YAML in `shared/taxonomy/` is authoritative. Regenerate human mapping
tables with:

```text
python -m floodsight_data.cli taxonomy --write-tables
```

Mappings remain candidates until complete real datasets are inventoried and reviewed.

## Segmentation taxonomy v1

| ID | Class | Source support |
| ---: | --- | --- |
| 0 | `background_other` | FloodNet, RescueNet |
| 1 | `water` | Both; pool merges but does not imply flooding |
| 2 | `road_clear` | FloodNet non-flooded road; RescueNet road-clear |
| 3 | `road_flooded` | FloodNet only |
| 4 | `road_blocked` | RescueNet only |
| 5 | `building_normal` | Both |
| 6 | `building_flooded` | FloodNet only |
| 7 | `building_minor_damage` | RescueNet only |
| 8 | `building_major_damage` | RescueNet only |
| 9 | `building_destroyed` | RescueNet only |
| 10 | `vehicle` | Both |
| 11 | `vegetation` | Both; supported tree/grass merge |

Ignore index is `255`. Debris/landslide is a reserved product concept, not a v1
trainable class. `road_flooded` and `road_blocked` are never merged.

## Detection taxonomy v1

| Target ID | Target | VisDrone source IDs |
| ---: | --- | --- |
| 0 | `person` | 1 pedestrian + 2 people |
| 1 | `car` | 4 |
| 2 | `van` | 5 |
| 3 | `truck` | 6 |
| 4 | `bus` | 9 |
| 5 | `bicycle` | 3 |
| 6 | `motorcycle` | 10 motor |
| 7 | `tricycle` | 7 tricycle + 8 awning-tricycle |

VisDrone 0 ignored-region and 11 others are explicitly ignored and never counted.

## Semantic compromises

- Non-flooded building maps to `building_normal`; it is not a damage inspection.
- Non-flooded road maps to `road_clear`; later accessibility still needs evidence.
- Pool merges to `water` but must not be interpreted as flood exposure.
- Tree and grass merge into `vegetation`.
- Several target classes have only one source, so provenance must remain in loaders
  and evaluation.
