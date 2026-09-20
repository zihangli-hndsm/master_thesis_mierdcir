# Intent coverage v2: automatic descriptive scan

This is a rule-based descriptive scan, not ground truth. Human intent
validation was intentionally skipped in this release because annotators were
unavailable; no category-level or causal claim is made from these labels.

| Source | Rows | Automatic labels | Manual review target |
|---|---:|---|---:|
| MTCIR_test_raw | 10,000 | {'spatial_positional': 1286, 'negative_constraint': 7097, 'comparative_intensity': 158, 'global_view': 334, 'instance_level': 1073, 'functional_affordance': 52} | 100 |
| MerdCIR_test | 5,000 | {'instance_level': 1377, 'negative_constraint': 1140, 'spatial_positional': 1220, 'comparative_intensity': 518, 'global_view': 542, 'functional_affordance': 203} | 100 |
| CIRR_val | 4,181 | {'instance_level': 2232, 'negative_constraint': 374, 'comparative_intensity': 412, 'functional_affordance': 41, 'spatial_positional': 1016, 'global_view': 106} | 100 |
| FashionIQ_val | 6,016 | {'comparative_intensity': 3628, 'global_view': 187, 'instance_level': 1334, 'spatial_positional': 218, 'negative_constraint': 648, 'functional_affordance': 1} | 100 |

The emitted manual sheet contains blank `manual_intent`, `manual_valid`, and
`manual_notes` columns. It is retained as an audit artifact, while the
automatic distributions are reported as descriptive evidence only.

Pairwise benchmark Jensen–Shannon divergences, CIRR per-intent performance,
and the exploratory frequency–performance coefficients are reported in
`intent_distribution_analysis_v2.md` and the corresponding v2 metric CSVs.
