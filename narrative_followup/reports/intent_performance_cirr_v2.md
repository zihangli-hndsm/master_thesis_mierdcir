# CIRR per-intent performance v2

These hit rates use automatic lexical labels and frozen prediction exports. They
are descriptive only. Manual intent validation was intentionally skipped in
this release because annotators were unavailable; no intent-specific mechanism
claim is made.

`n` counts query-seed observations (4,181 queries × 3 seeds). The metric is the first-ranked target hit rate for the corresponding global or candidate-subset export.

| Condition | Metric | Automatic intent | n | Hit@1 |
|---|---|---|---:|---:|
| fixed | recall | comparative_intensity | 1236 | 0.2047 |
| fixed | recall | functional_affordance | 123 | 0.2846 |
| fixed | recall | global_view | 318 | 0.3019 |
| fixed | recall | instance_level | 6696 | 0.2857 |
| fixed | recall | negative_constraint | 1122 | 0.1381 |
| fixed | recall | spatial_positional | 3048 | 0.2326 |
| fixed | recall_subset | comparative_intensity | 1236 | 0.6570 |
| fixed | recall_subset | functional_affordance | 123 | 0.7317 |
| fixed | recall_subset | global_view | 318 | 0.7107 |
| fixed | recall_subset | instance_level | 6696 | 0.6850 |
| fixed | recall_subset | negative_constraint | 1122 | 0.5285 |
| fixed | recall_subset | spatial_positional | 3048 | 0.6220 |
| multi | recall | comparative_intensity | 1236 | 0.2071 |
| multi | recall | functional_affordance | 123 | 0.2358 |
| multi | recall | global_view | 318 | 0.2925 |
| multi | recall | instance_level | 6696 | 0.2842 |
| multi | recall | negative_constraint | 1122 | 0.1845 |
| multi | recall | spatial_positional | 3048 | 0.2480 |
| multi | recall_subset | comparative_intensity | 1236 | 0.6303 |
| multi | recall_subset | functional_affordance | 123 | 0.6748 |
| multi | recall_subset | global_view | 318 | 0.6761 |
| multi | recall_subset | instance_level | 6696 | 0.6644 |
| multi | recall_subset | negative_constraint | 1122 | 0.5793 |
| multi | recall_subset | spatial_positional | 3048 | 0.5965 |

The manual review sheet was intentionally not completed in this release; these
automatic labels must not be treated as human-validated categories.
