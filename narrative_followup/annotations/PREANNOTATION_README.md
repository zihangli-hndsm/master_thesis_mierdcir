# Mechanism preannotation handoff

`cirr_mechanism_preannotations_v1.csv` is a caption-only, rule-based review
aid for the 600-row mechanism candidate queue. It is deliberately separate
from `cirr_mechanism_candidates.csv` and must not be treated as human labels.

The suggestions use lexical rules for possible mechanism terms and operation
verbs. They do not inspect the reference or target image, cannot establish
grounding, and may list a low-confidence primary label when several rules
fire. Every row has `review_required=yes`.

Human annotators should use the caption and contact sheets to replace or
reject the suggestions in the human annotation sheet. The preannotations are
not used for category-level statistics, agreement, or causal claims.

Generated locally on 2026-08-26 from the frozen 600-row candidate sheet.
