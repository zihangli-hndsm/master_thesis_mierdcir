# ViT-L Capacity Pilot Report (matched-step, exploratory)

**Date:** 2026-08-14
**Status:** draft — numbers pending 2×2 evaluation
**Protocol:** reduced matched-step 2400 (§8.3/§8.4 as adapted 2026-08-14; full 3-epoch protocol infeasible in 30h budget)

## 1. Protocol (as run)

| Condition | Backbone | Batch | Steps | Seed | Data |
|-----------|----------|-------|-------|------|------|
| Fixed-L | ViT-L/14 | 96 | 2400 (epoch 0, interrupted) | 42 | pair_control/train_fixed.jsonl |
| Multi-L | ViT-L/14 | 96 | 2400 | 42 | pair_control/train_multi.jsonl |
| Fixed-B96 | ViT-B/32 | 96 | 2400 | 42 | pair_control/train_fixed.jsonl |
| Multi-B96 | ViT-B/32 | 96 | 2400 | 42 | pair_control/train_multi.jsonl |

- Batch 96 matched across backbones (in-batch negative count identical; §8.4).
- Validation every 300 steps + epoch end on common-dev (pair_control/dev.jsonl).
- Checkpoint selection: step 2400 snapshot per condition (Fixed-L uses the backed-up `step2400.pth.tar`; others from topk jsons).
- CIRR val validation skipped during training (`--skip_cirr_validation`); evaluated post-hoc.
- **Limitations:** 1 epoch vs the ViT-B main matrix (3 epochs, batch 300); results are exploratory, not a formal capacity test. Absolute scores are NOT comparable to the pair-control main results.

## 2. Checkpoints

| Condition | Path | SHA256 (TBD) |
|-----------|------|---------------|
| Fixed-L | checkpoints/topk_pair_fixed_L_s42/step2400.pth.tar | |
| Multi-L | checkpoints/topk_pair_multi_L_s42/<step2400> | |
| Fixed-B96 | checkpoints/topk_pair_fixed_B96_s42/<step2400> | |
| Multi-B96 | checkpoints/topk_pair_multi_B96_s42/<step2400> | |

## 3. Results (completed 2026-08-15)

| Dataset | Metric | Fixed-L | Multi-L | Fixed-B96 | Multi-B96 |
|---------|--------|---------|---------|-----------|-----------|
| MTCIR (exact) | R@1 / mAP | 0.4636 / 0.5863 | 0.4420 / 0.5768 | 0.3683 / 0.4942 | 0.3361 / 0.4727 |
| MerdCIR (exact) | R@1 / mAP | 0.5396 / 0.6535 | 0.5538 / 0.6765 | 0.4556 / 0.5796 | 0.4738 / 0.6040 |
| FashionIQ | R@1 (avg) | 0.0559 | 0.0475 | 0.0437 | 0.0342 |
| CIRR val | recall R@1 / R@5 | 0.1796 / 0.4248 | 0.1516 / 0.3853 | 0.1799 / 0.4229 | 0.1629 / 0.3975 |
| CIRR val | subset R@1 / R@3 | 0.5551 / 0.8795 | 0.5068 / 0.8598 | 0.5597 / 0.8850 | 0.5260 / 0.8708 |

All checkpoints: seed 42, batch 96, step 2400 (epoch 0). Eval logs: `logs/vitl_2x2_eval_20260815_022504.log`, `logs/vitl_2x2_cirr_20260815_032704.log`, `logs/vitl_2x2_subset_20260815_040000.log`.

## 4. Capacity interaction (§8.2)

\[
\Delta_L = \mathrm{Multi\text{-}L} - \mathrm{Fixed\text{-}L}, \quad
\Delta_B = \mathrm{Multi\text{-}B96} - \mathrm{Fixed\text{-}B96}
\]

| Metric | Δ_L | Δ_B | Δ_L − Δ_B | Supports capacity bottleneck? |
|--------|-----|-----|-----------|------------------------------|
| MTCIR R@1 | −0.0216 | −0.0322 | +0.0106 | weak (direction only) |
| MerdCIR R@1 | +0.0142 | +0.0182 | −0.0040 | no |
| FashionIQ R@1 | −0.0084 | −0.0095 | +0.0011 | no |
| CIRR recall R@1 | −0.0280 | −0.0170 | −0.0110 | no |
| CIRR subset R@1 | −0.0483 | −0.0337 | −0.0146 | no |

Interpretation: On the target domain (CIRR), Multi is below Fixed on BOTH backbones, and the gap is LARGER on ViT-L than on ViT-B (Δ_L − Δ_B < 0). Backbone scaling improves source-domain scores (MTCIR 0.46 vs 0.37; MerdCIR 0.54 vs 0.46) but does NOT release cross-domain multi-intent gains. No support for the capacity-bottleneck hypothesis in this reduced protocol.

## 5. §8.7 trigger decision

Trigger (pre-registered): Multi-L vs Fixed-L on CIRR full R@1/R@5 shows substantial improvement, RecallSubset no longer clearly negative, and Δ_L − Δ_B > 0.

**Decision (2026-08-15): DO NOT extend to seeds 123/2025; DO NOT submit ViT-L models to CIRR test1.**
- Multi-L CIRR recall R@1 (0.152) is BELOW Fixed-L (0.180), −2.8pp (no improvement)
- Multi-L subset R@1 (0.507) is below Fixed-L (0.555), −4.8pp (negative gap, not reduced)
- Δ_L − Δ_B < 0 on CIRR recall and subset
- → All three pre-registered conditions fail. Submit the 9 frozen ViT-B checkpoints only.

Caveat: this pilot is 1-epoch (2400 steps) + batch 96 + single seed. The main matrix (3 epochs, batch 300) shows MULTI > FIXED on CIRR recall (+0.66pp); the pilot's opposite sign likely reflects under-training (multi-intent supervision needs more updates to pay off). Absolute pilot numbers are NOT comparable to the main matrix.

## 6. Notes

- Fixed-L original 3-epoch run was interrupted by SLURM job timeout at step 2400/7983 (2026-08-13 19:43); resumed protocol uses the step-2400 snapshot.
- Training speed (ViT-L, batch 96, single GPU): 3.08 s/it; validation ~18 min per 300-step point.
