# CIRR mechanism annotation protocol

The 600-row `cirr_mechanism_candidates.csv` is a queue for human review. The
100-row `cirr_mechanism_overlap_100.csv` is the balanced double-annotation
subset: 25 rows from each candidate group. Both annotators must label the
overlap independently before comparing answers.

The candidate groups are selected from frozen three-seed rank deltas:

- `multi_global_win`: 150 pairs with positive global delta for all three seeds;
- `multi_subset_loss`: 150 pairs with negative subset delta for all three seeds;
- `fixed_win`: 100 pairs with negative global delta for all three seeds;
- `stratified_random`: 200 remaining pairs sampled round-robin across global/subset sign strata.

The sign convention is `delta = rank_Fixed - rank_Multi`: positive means Multi
places the target earlier, while negative subset delta means Fixed places it
earlier within the candidate set. The exact source hashes and selected IDs are
recorded in `cirr_mechanism_selection_manifest.json`.

## Primary mechanism

Choose exactly one label:

`attribute_style`, `entity_replacement`, `count`, `spatial_relation`,
`negation_removal`, `scene_context`, `function_use_case`,
`comparative_intensity`, `omission`, `intent_drift`, `entity_binding`,
`operation_polarity`, `none`, or `ambiguous`.

Use the caption and the reference/target pair together. `none` means the
requested change is not visibly supported or cannot be assigned to a listed
mechanism. `ambiguous` means the evidence is visible but two or more primary
labels are equally plausible.

## Additional fields

- `additional_tags`: zero or more of the remaining mechanism labels, separated
  by `|`.
- `grounding`: `global`, `local`, `both`, or `unclear`.
- `operation_polarity`: `add`, `remove`, `replace`, `move`, `intensify`,
  `reduce`, `compare`, `none`, or `unclear`.
- `notes`: short evidence-based justification; do not infer the model's cause.

The global/subset winner and rank fields are computed facts from frozen
predictions, not annotation decisions. Agreement should be computed on the
primary mechanism and separately on grounding/polarity. Do not report
category-specific mechanism statistics until the labels are filled, checked,
and agreement is reported.
