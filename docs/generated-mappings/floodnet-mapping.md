# floodnet mapping floodnet-mapping-v1

| Source ID | Source class | Action | Target ID | Target class | Review | Explanation |
| ---: | --- | --- | ---: | --- | --- | --- |
| 0 | background | MAP | 0 | background_other | REVIEW_REQUIRED | Background and unmodelled scene content. |
| 1 | building_flooded | MAP | 6 | building_flooded | REVIEW_REQUIRED | Direct flooded-building support. |
| 2 | building_non_flooded | MAP | 5 | building_normal | REVIEW_REQUIRED | Non-flooded is the supported normal-building state; it does not imply damage inspection. |
| 3 | road_flooded | MAP | 3 | road_flooded | REVIEW_REQUIRED | Flooded road remains distinct from physical blockage. |
| 4 | road_non_flooded | MAP | 2 | road_clear | REVIEW_REQUIRED | Non-flooded supports clear-of-flood only; operational traversability still needs later evidence. |
| 5 | water | MAP | 1 | water | REVIEW_REQUIRED | Direct water support. |
| 6 | tree | MERGE | 11 | vegetation | REVIEW_REQUIRED | Tree is merged into the broader vegetation training class. |
| 7 | vehicle | MAP | 10 | vehicle | REVIEW_REQUIRED | Direct semantic vehicle support. |
| 8 | pool | MERGE | 1 | water | REVIEW_REQUIRED | Pool is merged into water; it is not evidence of flooding. |
| 9 | grass | MERGE | 11 | vegetation | REVIEW_REQUIRED | Grass is merged into vegetation. |

> This is a Phase 3 code-ready candidate. Real source files and palettes must be inventoried before data-verified status.
