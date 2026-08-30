# floodnet mapping floodnet-mapping-v1

| Source ID | Source class | Action | Target ID | Target class | Review | Explanation |
| ---: | --- | --- | ---: | --- | --- | --- |
| 0 | background | MAP | 0 | background_other | SOURCE_FILE_CONFIRMED | Background and unmodelled scene content. |
| 1 | building_flooded | MAP | 6 | building_flooded | SOURCE_FILE_CONFIRMED | Direct flooded-building support. |
| 2 | building_non_flooded | MAP | 5 | building_normal | SOURCE_FILE_CONFIRMED | Non-flooded is the supported normal-building state; it does not imply damage inspection. |
| 3 | road_flooded | MAP | 3 | road_flooded | SOURCE_FILE_CONFIRMED | Flooded road remains distinct from physical blockage. |
| 4 | road_non_flooded | MAP | 2 | road_clear | SOURCE_FILE_CONFIRMED | Non-flooded supports clear-of-flood only; operational traversability still needs later evidence. |
| 5 | water | MAP | 1 | water | SOURCE_FILE_CONFIRMED | Direct water support. |
| 6 | tree | MERGE | 11 | vegetation | SOURCE_FILE_CONFIRMED | Tree is merged into the broader vegetation training class. |
| 7 | vehicle | MAP | 10 | vehicle | SOURCE_FILE_CONFIRMED | Direct semantic vehicle support. |
| 8 | pool | MERGE | 1 | water | SOURCE_FILE_CONFIRMED | Pool is merged into water; it is not evidence of flooding. |
| 9 | grass | MERGE | 11 | vegetation | SOURCE_FILE_CONFIRMED | Grass is merged into vegetation. |

> Source evidence is recorded per row. Human mapping, license, and visual review remain fingerprint-bound requirements before data-verified status.
