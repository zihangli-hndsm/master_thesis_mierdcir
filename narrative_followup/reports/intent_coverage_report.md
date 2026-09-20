# Intent coverage and text-surface audit

This report distinguishes the labeled exploratory VLM corpus from the deterministic Surface-v2 control. The latter is intentionally not assigned semantic intent labels.

## Exploratory VLM corpus

- Rows: **50,000**
- Frozen matched IDs covered: **50,000**
- JS divergence from uniform scenario coverage: **0.000016 bits**

| Scenario | Count | Proportion |
|---|---:|---:|
| surface_diverse_1 | 8,263 | 0.1653 |
| surface_diverse_2 | 8,436 | 0.1687 |
| surface_diverse_3 | 8,201 | 0.1640 |
| surface_diverse_4 | 8,355 | 0.1671 |
| surface_diverse_5 | 8,387 | 0.1677 |
| surface_diverse_6 | 8,358 | 0.1672 |

## Deterministic Surface-v2 frame balance

| Frame | Count |
|---|---:|
| 1 | 8,270 |
| 2 | 8,241 |
| 3 | 8,409 |
| 4 | 8,491 |
| 5 | 8,373 |
| 6 | 8,216 |

## Matched 50K text statistics

| Condition | Rows | Mean chars | Mean CLIP tokens | Mean NPs |
|---|---:|---:|---:|---:|
| fixed | 50,000 | 85.91 | 19.00 | 4.01 |
| surface | 50,000 | 120.77 | 26.58 | 4.01 |
| multi | 50,000 | 98.51 | 21.63 | 4.14 |

Interpretation: the exploratory corpus has near-balanced labeled scenario coverage, while the formal Surface-50K control isolates deterministic surface framing around the Fixed semantic core. A semantic intent claim still requires a common rubric and manual validation across benchmark query styles.

Structural rewrite-output checks for the 50K source artifact (including valid,
empty/non-text, malformed, and scenario-assignment counts) are in
`intent_generation_quality_v2.md` and
`metrics/intent_generation_quality_v2.json`.
