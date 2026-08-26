# visdrone_det mapping visdrone-mapping-v1

| Source ID | Source class | Action | Target ID | Target class | Review | Explanation |
| ---: | --- | --- | ---: | --- | --- | --- |
| 0 | ignored_region | IGNORE |  |  | SPEC_VERIFIED | Evaluation ignored region; never count as a person or vehicle. |
| 1 | pedestrian | MERGE | 0 | person | SPEC_VERIFIED | Pedestrian and people train the shared person class. |
| 2 | people | MERGE | 0 | person | SPEC_VERIFIED | People and pedestrian train the shared person class. |
| 3 | bicycle | MAP | 5 | bicycle | SPEC_VERIFIED | Retained as an individual aerial vehicle class. |
| 4 | car | MAP | 1 | car | SPEC_VERIFIED | Direct car support. |
| 5 | van | MAP | 2 | van | SPEC_VERIFIED | Direct van support. |
| 6 | truck | MAP | 3 | truck | SPEC_VERIFIED | Direct truck support. |
| 7 | tricycle | MERGE | 7 | tricycle | SPEC_VERIFIED | Tricycle and awning-tricycle share one target class. |
| 8 | awning_tricycle | MERGE | 7 | tricycle | SPEC_VERIFIED | Merged with tricycle while source provenance remains in the manifest. |
| 9 | bus | MAP | 4 | bus | SPEC_VERIFIED | Direct bus support. |
| 10 | motor | MAP | 6 | motorcycle | SPEC_VERIFIED | VisDrone motor maps to motorcycle. |
| 11 | others | IGNORE |  |  | SPEC_VERIFIED | Undefined/other objects are not a coherent training class. |

> This is a Phase 3 code-ready candidate. Real source files and palettes must be inventoried before data-verified status.
