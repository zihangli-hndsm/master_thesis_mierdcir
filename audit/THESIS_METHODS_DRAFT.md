# Thesis Methods Draft: Evaluation & Protocol (2026-08-13)

Draft sections for the thesis describing the evaluation pipeline as audited.
Status: draft for review; numbers/paths reference the audited state.

## 1. Evaluation Protocol

### 1.1 MTCIR / MerdCIR (source-domain)

- Evaluation split: frozen `pair_control/test.jsonl` (10,000 queries, 9,441 unique
  target images; split created 2026-08-08, seed 114514, triplet-level random shuffle).
- Gallery: all unique target images of the eval split; **global-gallery ranking**
  (every query ranked against the full gallery, never batch-local).
- Scoring: exact cosine via `eval_checkpoint.py --exact-gallery` (matmul, no ANN).
  Verified **exactly batch-invariant**: R@1 = 0.605800 at batch 64/128/256.
  The ChromaDB/HNSW path carries ≤ 2e-4 ANN noise and is not used for final numbers.
- Metrics: Recall@1/5/10/50, mAP (1/rank).
- Checkpoint selection: pre-registered, `c* = argmax common-dev mAP` per condition
  (common-dev = original MTCIR text on `pair_control/dev.jsonl`); CIRR val numbers
  reported from the selected checkpoints only.

### 1.2 Split hygiene (important caveat)

- Triplet-disjoint: 0 triplet overlap across train/dev/test.
- **Not image-disjoint**: 62.9% of test references and 58.6% of test targets also
  appear in train (any-role overlap: 14,018 images; train→test reference 5,366;
  train→test target 4,271). Reported numbers are therefore **pair-held-out**
  performance, not image-held-out generalization. (audit: `audit/MTCIR_AUDIT_REPORT.md`)
- A 100-query manual check confirmed query/reference/target/gallery-index mapping
  (`audit/mapping_spotcheck_table.md`); metric unit tests 6/6 pass
  (`audit/mtcir_data_audit.json`).

### 1.3 CIRR

- Val (dev-transfer): `cap.rc2.val.json` (4,181 queries). Two protocols:
  - Recall: global ranking over the val gallery; target = `target_hard`.
  - RecallSubset: ranking within `img_set.members`; target = `target_hard`.
- Test1 (official server): export `{"version":"rc2","metric":..., <pairid>: [...]}`
  with top-50 (recall) / top-3 (recall_subset); gallery = `split.rc2.test1.json`
  (2,315 images). Validated by `audit/cirr_validate.py`; rehearsal on val showed
  the exporter matches the local evaluator at float level and is invariant to
  batch size and query order at the metric level (GPU fp noise only reorders
  ties inside top-50 without affecting any Recall@K).
- Submission policy: 3 seeds × 3 conditions (RAW/FIXED/MULTI), mean ± std over
  seeds reported; no seed selection based on test scores
  (`cirr_test_results/SUBMISSION_GUIDE.md`).

### 1.4 FashionIQ

- Val split, three categories (dress/shirt/toptee), global gallery ranking,
  target-excluded, exact matmul scoring (identical code path as CIRR).

## 2. Model / Training Protocol

- ScheiCIR `cross_attn_alpha` (CrossAttentionBlock ×4 + AlphaGenerator MLP),
  CLIP ViT-B/32 backbone, InfoNCE (batch as negatives incl. reference as hard
  negative) + λ=30 soft-contrastive NP loss (temperature 0.07).
- 3-stage warmup: head (interaction/pooler/alpha, 1e-4) → +last encoder layers
  (1e-6) → joint (all, 1e-6), `joint_start_step=300`, 3 epochs, batch 300.
- Data: 255,400 strict-intersection triplets; text variants RAW (original MTCIR
  diff-lists), FIXED (single natural-language prompt), MULTI (6 intent scenarios).
- Seeds {42, 123, 2025}; fixed seed protocol.

## 3. ViT-L Capacity Pilot (§8 of the plan)

- Question: does ViT-B capacity limit the multi-intent generalization gain?
  Requires Δ_L − Δ_B > 0, where Δ = Metric(Multi) − Metric(Fixed) per backbone.
- Design: single-seed 2×2 (Fixed/Multi × ViT-B/ViT-L), same 255,400 pairs,
  same optimizer protocol, **matched in-batch negatives** (per-forward batch 96
  for both backbones; ViT-B batch-300 runs are re-trained at batch 96).
- Memory engineering: gradient checkpointing for ViT-L/H (`use_checkpoint`),
  segment-wise checkpointing of target/ref encoding, checkpoint disabled during
  validation (eval re-computation is 4× slower otherwise); batch 96 peaks at
  61.2 GB on A100-80GB.
- Decision rule (pre-registered): extend to seeds 123/2025 only if Multi-L shows
  a substantive CIRR full-R@1/R@5 gain over Fixed-L, RecallSubset no longer
  negative vs ViT-B, and Δ_L − Δ_B > 0.

## 4. ConText-CIR Text CC (reproduction evidence, §8.8)

- Loss-level scan (63 samples): raw CC ≈ 1.1e-4, ε=0.05 zeroes 100% → weighted
  λ·CC ≈ 9e-6 (already on record).
- Gradient probe (this session): ρ_g = ‖∇(λL_cc)‖/‖∇L_main‖ ≈ 2.5e-5 (untrained)
  and ≤ 1e-4 (converged λ_cc=0.08 checkpoint) on fusion/shared params;
  cosine(main, λ·cc) ≈ 0 (orthogonal); **ε>0 ⇒ L_cc ≡ 0 ⇒ zero gradient** on all
  early/mid/late batches.
- Claim (bounded): under our public-data reproduction and released/local
  implementation, the thresholded Text CC term did not enter the optimization
  objective; raw-CC gradient share is negligible. This does not refute
  ConText-CIR under its full training protocol.
