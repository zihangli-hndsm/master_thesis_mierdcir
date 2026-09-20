# CIRR automatic mechanism-oriented analysis v2

This is an automatic stratification and annotation-triage report. The
text fields are lexical proxies, not entity parsing or human mechanism
labels. No category-level causal claim is supported by this report.

## Rank-delta relationship

Rows: **12543** query-seed observations (4,181 queries × 3 seeds).
Positive delta means Multi places the target earlier under either mode.
For the binary analysis, global improvement is `delta_global > 0` and
subset degradation is `delta_subset < 0` (Fixed places the target earlier
within the subset).

| Comparison | n | Pearson r | Spearman r |
|---|---:|---:|---:|
| continuous_delta_global_vs_delta_subset | 12543 | 0.3550 | 0.3878 |
| global_improvement_vs_subset_degradation | 12543 | -0.1524 | -0.1524 |

## Outcome counts and text proxies

`entity_mention_proxy_count` counts determiner/number-led lexical spans;
it is not a named-entity or noun-phrase parser. `negation_proxy` and
`entity_binding_proxy` are binary lexical indicators.

| Group | n | mean words | mean chars | mean entity proxy | negation proxy | binding proxy |
|---|---:|---:|---:|---:|---:|---:|
| all | 12543 | 11.01 | 58.22 | 1.72 | 0.0899 | 0.1222 |
| global_multi_win | 4165 | 10.82 | 57.20 | 1.73 | 0.1263 | 0.1273 |
| global_fixed_win | 3479 | 11.08 | 58.61 | 1.70 | 0.0799 | 0.1158 |
| subset_multi_win | 1205 | 10.72 | 56.64 | 1.69 | 0.1834 | 0.1419 |
| subset_fixed_win | 1476 | 11.07 | 58.91 | 1.60 | 0.0840 | 0.1240 |
| global_win_subset_loss | 200 | 10.71 | 56.55 | 1.62 | 0.1400 | 0.1100 |

## Extreme rows for human review

The full rows are in `cirr_tradeoff_extremes_v2.csv`, with captions and
frozen ranks. Each extreme type contains unique pair IDs and keeps the
most extreme seed-level observation for each pair. These are triage samples only; reference/target contact
sheets and human labels remain authoritative.

| Extreme type | Rows exported | Selection rule |
|---|---:|---|
| global_multi_win | 10 | largest positive global delta |
| global_fixed_win | 10 | most negative global delta |
| subset_multi_win | 10 | largest positive subset delta |
| subset_fixed_win | 10 | most negative subset delta |

Machine-readable outputs:

- `cirr_tradeoff_features_per_seed.csv`
- `cirr_tradeoff_text_feature_summary_v2.csv`
- `cirr_tradeoff_correlations_v2.csv`
- `cirr_tradeoff_extremes_v2.csv`
- `cirr_tradeoff_bootstrap_ci_v2.json`

Figures generated from these automatic outputs are
`figures/global_subset_tradeoff.pdf`, `figures/intent_win_loss_enrichment.pdf`,
and `figures/train_test_intent_distribution.pdf`. The last figure explicitly
marks the training scenario and benchmark taxonomies as non-equivalent.

The bootstrap is paired at the query/pair level: the three seed rows
for each pair are resampled together. It quantifies query-level
uncertainty and does not replace separate seed variability or human
annotation.

All outputs are automatic-only and must not be used to replace the
planned human mechanism annotation and double-annotation agreement.
