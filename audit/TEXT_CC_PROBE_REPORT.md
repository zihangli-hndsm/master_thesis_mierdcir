# Text CC Matched-Gradient Probe Report

**Date:** 2026-08-13
**Plan reference:** `NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md` §8.8
**Code:** `reproduction_experiments/ConText-CIR-09052026/reproduction_evidence/text_cc_gradient_probe.py`
**Model:** ConText-CIR local reproduction (ViT-B/32, Config-B residual fusion, λ_cc=0.08), untrained init
**Data:** CIRR train split (28,225), early/mid/late regions (8 samples each, max_nps=3)

## Protocol (per §8.8)

For each (stage, epsilon) cell, with identical init/batch/forward outputs:
1. backward `L_main` (InfoNCE), unweighted `L_cc`, and `λ·L_cc` separately
2. gradient norms over **fusion/shared params** (`cross_attn_layers` + `attn_pooler`)
3. `ρ_g = ‖∇(λ·L_cc)‖ / ‖∇L_main‖` and cosine between the two gradients
4. epsilon ∈ {0.0, 0.05 (Train.py default), 0.08}

## Results

| stage | ε | loss_main | loss_cc | **ρ_g** | cosine(main, λ·cc) |
|---|---|---|---|---|---|
| early | 0.0 | 2.7858 | 1.27e-4 | **3.38e-05** | +0.0048 |
| early | 0.05 | 2.7858 | 0.0 | **0** | nan (zero grad) |
| early | 0.08 | 2.7858 | 0.0 | **0** | nan |
| mid | 0.0 | 2.8452 | 1.12e-4 | **2.53e-05** | +0.0008 |
| mid | 0.05 | 2.8452 | 0.0 | **0** | nan |
| mid | 0.08 | 2.8452 | 0.0 | **0** | nan |
| late | 0.0 | 2.9102 | 1.05e-4 | **2.81e-05** | −0.0035 |
| late | 0.05 | 2.9102 | 0.0 | **0** | nan |
| late | 0.08 | 2.9102 | 0.0 | **0** | nan |

Raw results: `reproduction_evidence/text_cc_gradient_probe.json`.

## Interpretation (bounded to the local released-code reproduction)

1. **Thresholded Text CC is inactive under any ε>0**: with ε=0.05 (paper/Train.py default) and ε=0.08, `loss_cc ≡ 0` on every stage batch → **zero gradient, the term does not enter the optimization objective** in this setting. This extends the earlier loss-level scan (63/63 samples zeroed, `text_cc_epsilon_scan_64_summary.json`) to the gradient level.
2. **Even with ε=0 (raw CC), the gradient contribution is negligible**: ρ_g ≈ 2.5–3.4e-5, i.e. five orders of magnitude below the main contrastive gradient on the shared fusion parameters. A loss scalar at ~1e-4 with λ=0.08 (~1e-5 weighted) is consistent with this tiny gradient share.
3. **Gradient direction is uncorrelated with the main objective**: cosine(main, λ·cc) ≈ ±0.005 (≈orthogonal) across stages.
4. **No stage dependence**: early/mid/late data regions give the same picture.

## Caveats

- Probe uses the **untrained** ConText init (training-position approximation via data regions). The ε>0 ⇒ loss≡0 conclusion is init-independent (it only requires |attn_in_context − attn_isolated| < ε, observed on all sampled captions); the ρ_g magnitude could shift during training but the scan + this probe jointly show no regime where the term is non-negligible.
- This **does not refute ConText-CIR** (no Aggregated data, official checkpoints, or full training protocol); per §8.9 the claim is bounded to:
  > Under our bounded public-data reproduction and released/local implementation, the thresholded Text CC signal was inactive or negligible.
