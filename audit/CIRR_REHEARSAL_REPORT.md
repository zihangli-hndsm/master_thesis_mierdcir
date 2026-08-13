# CIRR Submission Pipeline Validation Report

**Date:** 2026-08-13
**Plan reference:** `NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md` §3-§5
**Status:** exporter + validator ready; val rehearsal PASSED; test1 export flow verified on one frozen checkpoint; **no test1 submission yet** (per §10, submission waits for the ViT-L pilot decision).

## 1. Exporter implementation (already in eval_checkpoint.py)

- `evaluate_cirr_and_dump(..., metric="recall")` → top-50 global ranking per query, reference excluded, `{"version":"rc2","metric":"recall", <pairid>: [50 ids]}`.
- `metric="recall_subset"` → top-3 within `img_set.members` (reference excluded), `{"metric":"recall_subset", ...}`.
- Global gallery from `split.rc2.test1.json` (2315 unique test1 images, all present in LMDB: 2315 ok / 0 missing).
- **Fixed 2026-08-13:** `_load_cirr_gallery_ids` now handles the dict format `{img_id: "./test1/xxx.png"}` (previously returned empty → gallery silently degraded to query-derived images only).

## 2. Validator (`audit/cirr_validate.py`)

Checks: parseability, version=rc2, metric∈{recall, recall_subset}, key coverage (no missing/extra/duplicate pairids), exactly 50/3 unique candidates, candidates ∈ test1 split, no reference in predictions, subset ⊆ img_set.members, file size < 5MB, SHA256 manifest of prediction + checkpoint + command.

## 3. Val rehearsal (§5) — checkpoint `topk_pair_multi_s42` E2 S600

| step | result |
|---|---|
| direct `evaluate_cirr_recall` (val) | R@1=0.259029, R@5=0.546281, R@10=0.667783, R@50=0.881846 |
| direct `evaluate_cirr_subset` (val) | sub@1=0.639082, sub@2=0.833293, sub@3=0.918441 |
| exported JSON → recompute (recall) | R@1=0.25902894 … identical to direct (float-level) PASS |
| exported JSON → recompute (subset) | sub@1=0.63908156 … identical to direct PASS |
| export batch 128 vs 64 (recall) | recomputed metrics identical PASS |
| query-order invariance (shuffled loader) | 153/4181 pairid-level top-50 order flips, **all Recall@K identical** (0.259029/…/0.881846) — flips are GPU fp nondeterminism (~1e-7 similarity ties) inside top-50, no metric impact. Test1 path (fixed dict gallery order + shuffle=False loader) is deterministic. |

**compute_cirr_metrics.py fixed 2026-08-13:** recall_subset now scores against `target_hard` when available (was: first img_set member → R@1 always 1 for top-3 predictions). `--gt` flag added for val rehearsal.

## 4. Test1 export smoke (single frozen checkpoint: multi s42)

- `cirr_test_results/raw_s42_recall.json` (4.19 MB) — validator **PASS** (all 11 checks)
- `cirr_test_results/raw_s42_recall_subset.json` (0.29 MB) — validator **PASS** (all 12 checks)
- manifests: `<file>.validation.json` (SHA256 + command)

## 5. Frozen checkpoint manifest (§2.1)

`cirr_test_results/checkpoint_manifest.json` — 9 checkpoints (RAW/FIXED/MULTI × s42/s123/s2025), all E2 S600, selected by common-dev mAP (pre-registered), SHA256 recorded, test split never used for selection.

## 6. Remaining before submission (next session)

1. ViT-L pilot decision (§8.7 trigger) → final frozen checkpoint set
2. Regenerate test1 recall + recall_subset JSONs for the final set (≈1-2 min/model)
3. Run validator on all files; record hashes in `submission_manifest.json`
4. Upload to official server (user action; account needed), record server results in `server_results.csv`
