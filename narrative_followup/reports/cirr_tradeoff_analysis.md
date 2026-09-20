# CIRR global/subset rank-delta analysis

## Scope

This post-hoc analysis uses the same three Raw, Fixed and Multi training seeds and the CIRR `rc2.val` captions. Fixed and Multi predictions were exported with the same frozen checkpoints and evaluator. Global exports retain top-50 predictions; subset exports retain top-3 predictions. Consequently, ranks are censored at 51 and 4 respectively when the target is not present.

## Fixed versus Multi

Across 4,181 CIRR val queries and three seeds (12,543 query-seed observations), the mean censored rank difference is:

| Mode | Multi better | Fixed better | Ties | Mean `(rank_Fixed - rank_Multi)` |
|---|---:|---:|---:|---:|
| Global | 4,165 | 3,479 | 4,899 | +1.088 |
| Subset | 1,205 | 1,476 | 9,862 | −0.027 |

Positive global delta means Multi places the target earlier. The sign reverses for subset ranking, although the subset mean is small and most queries tie because the candidate set has only three ranked positions. This is consistent with the aggregate CIRR result: Multi has an advantage in full-gallery coverage, while Fixed has a small candidate-set advantage.

## Interpretation boundary

This result strengthens the claim that the global/subset trade-off is query-level and reproducible across seeds. It does not identify which linguistic intent causes the trade-off. The predictions are top-k exports rather than full sorted galleries, and no human taxonomy has yet been applied. Therefore the current defensible wording is:

> Multi changes the distribution of rank outcomes in a metric-dependent way: it improves full-gallery target placement more often and by a larger average censored margin, while Fixed retains a small advantage in the restricted candidate set.

The stronger claim that scene/function intents explain global wins while attribute/entity/negation intents explain subset losses remains pending the next-session annotation phase.

Raw files:

- `metrics/cirr_val_rank_deltas_per_seed.csv`
- `metrics/cirr_val_rank_deltas_aggregated.csv`
- `metrics/cirr_val_rank_delta_summary.json`

The automatic text-proxy summaries, global/subset correlations, unique
extreme-pair triage rows, and query-level paired bootstrap are in
`cirr_mechanism_auto_analysis_v2.md` and the `cirr_tradeoff_*_v2` outputs.
They do not replace human annotation.
