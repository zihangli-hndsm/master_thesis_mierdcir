# Narrative follow-up：当前 claim assessment

## Completed evidence in this session

### Query-style matrix

The 27-cell matrix evaluates nine frozen checkpoints on the same 10,000 pair-held-out pairs and gallery. The query style is the only changed evaluation input.

At Recall@1, the best training condition changes with the query style:

| Query style | Raw model | Fixed model | Multi model | Best |
|---|---:|---:|---:|---|
| Raw | 0.6063 | 0.4943 | 0.5691 | Raw |
| Fixed | 0.7713 | 0.8254 | 0.7888 | Fixed |
| Multi | 0.5321 | 0.5082 | 0.5891 | Multi |

Thus the diagonal winner is consistent: Raw wins Raw queries, Fixed wins Fixed queries, and Multi wins Multi queries. However, Fixed queries are easier for all three models in absolute terms, so the result supports style-conditional model preference rather than a pure causal alignment effect. It does not isolate semantic intent diversity from rewrite fluency or surface regularization.

### CIRR rank delta

The global/subset rank-delta analysis reproduces the direction of the existing aggregate trade-off at query level: Multi is better on more global observations and has a positive mean Fixed-minus-Multi rank delta; Fixed is slightly better on subset observations and the subset mean delta is negative. Since ranks are censored top-50/top-3, this is mechanism-supporting evidence, not a full-rank causal explanation.

## Claims currently supported

1. **Condition/query-style interaction:** supported descriptively by the complete 3×3 matrix, with the caveat that query-style difficulty differs.
2. **Metric-dependent CIRR trade-off:** supported by hidden-test aggregate results and the new val query-level rank deltas.
3. **Multi is not universally better:** supported; the source-domain and candidate-set results still show clear boundaries.

## Claims excluded from this release

1. **Semantic intent diversity beyond surface diversity:** still not established. The formal matched control is now complete, but Multi does not reproduce the Surface condition and therefore does not provide a positive semantic-intent result.
2. **Specific intent mechanisms:** excluded from this release. Human mechanism labels, agreement, and category enrichment were intentionally skipped because annotators were unavailable; the automatic lexical analysis is descriptive only.
3. **Universal transfer/generalization:** not supported and should not be claimed.

## New matched-control evidence

The three-seed Fixed/Surface/Multi control is complete on the same 50K training IDs. Surface training is slightly specialized to Surface queries (R@1 0.5829 versus Fixed-trained 0.5642), while Multi does not win on Multi queries and is lower than Fixed on both CIRR global and subset Recall@1. This isolates a measurable framing/style effect, but does not establish semantic-intent causality. See `narrative_followup/reports/narrative_followup_final_report.md` for the full matrix.

## Recommended paper wording now

Use “intent-routed rewriting” or “condition/query-style interaction,” and describe the CIRR observation as a “global-coverage versus candidate-set discrimination trade-off.” Avoid “semantic intent modeling” and avoid attributing the trade-off to specific intent categories. Human annotation was skipped in this release.
