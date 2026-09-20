# Narrative follow-up Gate 0 protocol audit

Status: PASS

- Raw/Fixed/Multi test files contain 10,000 rows each with identical ordered `(id, reference, target)` keys.
- Train/dev/test pair IDs are disjoint.
- Existing Surface-Diverse training artifact contains 50,000 unique IDs, all in pair-control train and outside dev/test.
- Nine frozen checkpoints were loaded from the checkpoint manifest and SHA256 verified.
- Existing pair-control evaluator and checkpoint-selection manifest are retained; no checkpoint is reselected in this narrative analysis.

The Surface-Diverse artifact is an input for a future formal control, not a completed three-seed experiment in this session.
