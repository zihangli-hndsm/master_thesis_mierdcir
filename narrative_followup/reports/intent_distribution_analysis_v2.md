# Automatic intent distribution analysis v2

This report uses the same six-label lexical taxonomy as the automatic
coverage scan. It is descriptive only. Manual intent validation was intentionally
skipped in this release because annotators were unavailable; no category-level
or causal claim is made from these labels.

## Benchmark distributions

| Source | Rows | negative_constraint | spatial_positional | comparative_intensity | functional_affordance | global_view | instance_level | JS to uniform (bits) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MTCIR_test_raw | 10000 | 0.7097 | 0.1286 | 0.0158 | 0.0052 | 0.0334 | 0.1073 | 0.293622 |
| MerdCIR_test | 5000 | 0.2280 | 0.2440 | 0.1036 | 0.0406 | 0.1084 | 0.2754 | 0.058003 |
| CIRR_val | 4181 | 0.0895 | 0.2430 | 0.0985 | 0.0098 | 0.0254 | 0.5338 | 0.195953 |
| FashionIQ_val | 6016 | 0.1077 | 0.0362 | 0.6031 | 0.0002 | 0.0311 | 0.2217 | 0.254259 |

## Pairwise Jensen–Shannon divergence (bits)

| Left | Right | JS divergence |
|---|---|---:|
| MTCIR_test_raw | MTCIR_test_raw | 0.000000 |
| MTCIR_test_raw | MerdCIR_test | 0.186640 |
| MTCIR_test_raw | CIRR_val | 0.347053 |
| MTCIR_test_raw | FashionIQ_val | 0.471904 |
| MerdCIR_test | MTCIR_test_raw | 0.186640 |
| MerdCIR_test | MerdCIR_test | 0.000000 |
| MerdCIR_test | CIRR_val | 0.080218 |
| MerdCIR_test | FashionIQ_val | 0.257115 |
| CIRR_val | MTCIR_test_raw | 0.347053 |
| CIRR_val | MerdCIR_test | 0.080218 |
| CIRR_val | CIRR_val | 0.000000 |
| CIRR_val | FashionIQ_val | 0.260429 |
| FashionIQ_val | MTCIR_test_raw | 0.471904 |
| FashionIQ_val | MerdCIR_test | 0.257115 |
| FashionIQ_val | CIRR_val | 0.260429 |
| FashionIQ_val | FashionIQ_val | 0.000000 |

## CIRR automatic intent performance

Per-intent hit rates are copied from `intent_performance_cirr_v2.csv`;
each rate uses automatic labels and is not human validated.

| Condition | Metric | Intent | n | Hit@1 |
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

## Frequency–performance association

The correlation uses the six CIRR-val automatic intent proportions as
frequency and the corresponding six per-intent CIRR hit rates as performance.
With six points, these coefficients are exploratory summaries, not tests
of causality.

| Condition | Metric | Frequency source | Pearson r | Spearman r |
|---|---|---|---:|---:|
| fixed | recall | CIRR_val | 0.1736 | -0.0857 |
| fixed | recall_subset | CIRR_val | -0.0242 | -0.4857 |
| multi | recall | CIRR_val | 0.3975 | 0.1429 |
| multi | recall_subset | CIRR_val | 0.0105 | -0.4286 |
