# rescuenet mapping rescuenet-mapping-v1

| Source ID | Source class | Action | Target ID | Target class | Review | Explanation |
| ---: | --- | --- | ---: | --- | --- | --- |
| 0 | background | MAP | 0 | background_other | REVIEW_REQUIRED | Background and unmodelled content. |
| 1 | water | MAP | 1 | water | REVIEW_REQUIRED | Direct water support; it does not by itself label a road flooded. |
| 2 | building_no_damage | MAP | 5 | building_normal | REVIEW_REQUIRED | Direct no-damage building support. |
| 3 | building_minor_damage | MAP | 7 | building_minor_damage | REVIEW_REQUIRED | Direct minor-damage support. |
| 4 | building_major_damage | MAP | 8 | building_major_damage | REVIEW_REQUIRED | Direct major-damage support. |
| 5 | building_total_destruction | MAP | 9 | building_destroyed | REVIEW_REQUIRED | Total destruction maps to destroyed. |
| 6 | vehicle | MAP | 10 | vehicle | REVIEW_REQUIRED | Direct semantic vehicle support. |
| 7 | road_clear | MAP | 2 | road_clear | REVIEW_REQUIRED | Clear road remains distinct from flooded and blocked roads. |
| 8 | road_blocked | MAP | 4 | road_blocked | REVIEW_REQUIRED | Physical/damage blockage is not treated as road flooding. |
| 9 | tree | MERGE | 11 | vegetation | REVIEW_REQUIRED | Tree is merged into vegetation. |
| 10 | pool | MERGE | 1 | water | REVIEW_REQUIRED | Pool is merged into water and does not imply flood exposure. |

> Source evidence is recorded per row. Human mapping, license, and visual review remain fingerprint-bound requirements before data-verified status.
