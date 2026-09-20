# ConText-CIR complete results and evidence record

**Version:** 2026-08-27

This document consolidates the available ConText-CIR evidence into one result
record. It distinguishes paper-reported reference numbers, local validation,
local reconstructed training, and CIRR server submissions. A number is never
treated as a reproduction result merely because it appears in the paper.

## 1. Why the previous package looked incomplete

The previous release builder created a selective audit package. It copied
selected reports and compact metrics, but intentionally did not copy the full
`reproduction_evidence/` run tree, all session reports, or the original D0/D3
submission JSONs. The results existed locally, but their evidence trail was
distributed across:

- `ConText-CIR-09052026/reproduction_evidence/`;
- `ConText-CIR-09052026/public_audit_packet/`;
- `cirr_test_results/session_b_no_np/`.

This release adds the consolidated documents and the missing compact result
files to `context_cir_reproduction/`. Checkpoints remain external artifacts
with provenance records rather than being duplicated into the archive.

## 2. Paper-reported reference results

These are comparison references transcribed from the uploaded paper; they are
not local results.

### CIRR supervised setting (paper Table 1)

| Backbone | Training/data setting | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ViT-B or smaller | Aggregated | 49.83 | 81.54 | 89.76 | 98.95 | 76.64 | 90.69 | 96.31 |
| ViT-L | Aggregated | 52.65 | 83.27 | 89.51 | 98.87 | 80.32 | 92.13 | 96.08 |
| ViT-H | Aggregated | 55.24 | 84.85 | 90.75 | 98.82 | 82.96 | 93.12 | 97.04 |

### Dataset ablation (paper Table 4, ViT-H)

| Training data | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| CIRR | 45.25 | 77.52 | 86.88 | 97.24 |
| CIRR + CIRR_R | 48.54 | 80.12 | 88.52 | 97.49 |
| CIRR + CIRR_R + LaSCo | 53.12 | 83.78 | 89.48 | 98.67 |
| Aggregated | 55.24 | 84.85 | 90.75 | 98.82 |

### Text-CC ablation (paper Table 5, ViT-H, Aggregated)

| Setting | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| No Text-CC (`lambda=0`) | 48.92 | 79.86 | 88.74 | 97.41 |
| With Text-CC (`lambda=0.08`) | 55.24 | 84.85 | 90.75 | 98.82 |

The reported difference is **+6.32 percentage points at R@1**. The paper
reference uses ViT-H and Aggregated data; it must not be compared as if it
were the local ViT-B/CIRR-only experiment.

## 3. Local source and data audit

| Item | Result |
|---|---|
| Source compilation and dependency checks | Passed |
| CIRR train LMDB | 16,939 mapped images checked; missing = 0 |
| CIRR test1 LMDB | 2,315 mapped images checked; missing = 0 |
| FashionIQ validation LMDB | 14,975 entries; 440 download failures recorded |
| FashionIQ test LMDB | 14,917 entries; 500 download failures recorded |
| Main local converged data | Public CIRR only for the historical B/L/H runs |
| LaSCo-inclusive control data | CIRR + CIRR-R + LaSCo + Hotels-50K |

The paper's complete Aggregated recipe and author-provenance training setup
were not reproduced exactly.

## 4. Early bounded D0/D1/D2/D3 matrix

These are one-seed, three-epoch, ViT-B, residual-fusion, no-Text-CC local
validation results. D0 = CIRR; D1 = CIRR + CIRR-R; D2 = CIRR + Hotels-50K;
D3 = CIRR + CIRR-R + Hotels-50K.

| ID | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 5.19 | 19.76 | 31.05 | 66.75 | 30.71 | 53.67 | 72.66 | 23 |
| D1 | 5.53 | 22.32 | 34.78 | 70.34 | 25.90 | 48.79 | 68.48 | 20 |
| D2 | 5.60 | 20.16 | 31.12 | 63.69 | 27.53 | 50.18 | 69.60 | 26 |
| D3 | 8.13 | 26.45 | 39.70 | 75.51 | 28.80 | 50.83 | 70.82 | 17 |

The data mixture improves full-gallery retrieval in this run but not
candidate-set retrieval. It is not evidence for a uniform CIRR gain.

## 5. Historical local converged runs

These runs have different protocols and are therefore descriptive, not a
single controlled ablation.

| Run | Backbone | Data | Text-CC | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Public CIRR no-CC | ViT-B | CIRR | off | 3.95 | 17.89 | 30.59 | 66.71 | — | — | — |
| Public CIRR Text-CC, `epsilon=0.08` | ViT-B | CIRR | on | 5.12 | 36.79 | 53.41 | 86.18 | — | — | — |
| Session E no-CC | ViT-L | CIRR | off | 16.36 | 47.05 | 62.86 | 89.50 | 40.25 | 64.91 | 81.25 |
| Session E Text-CC positive control, 3 epochs | ViT-B | CIRR | on | 14.11 | 41.16 | 56.88 | 87.30 | 37.72 | 61.42 | 78.47 |
| Public CIRR no-NP approximation | ViT-H | CIRR | disabled | 10.67 | 46.16 | 64.55 | 91.41 | — | — | — |

The ViT-H row is a **no-NP/no-Text-CC approximation**, not the paper's
Text-CC-enabled ViT-H result. The Session E rows use separate run settings and
must not be used to claim a clean Text-CC effect.

## 6. Official CIRR test submissions for the local D0/D3 control

The following are actual CIRR server-returned results for two seeds. They are
official for the submitted local D0/D3 protocol, not official reproduction of
the paper's Aggregated recipe.

**Protocol:** ViT-B, residual fusion, no NP, `lambda_cc=0`; D0 = CIRR; D3 =
CIRR + CIRR-R + Hotels-50K; test results were not used for checkpoint
selection.

| Condition | Seed | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 4.819 | 18.458 | 29.157 | 64.675 | 31.976 | 54.699 | 73.542 |
| D0 | 1 | 5.205 | 19.470 | 31.108 | 66.747 | 32.458 | 54.843 | 73.735 |
| D3 | 0 | 8.602 | 26.819 | 40.699 | 74.337 | 29.205 | 52.988 | 71.831 |
| D3 | 1 | 7.831 | 26.072 | 39.133 | 72.892 | 28.747 | 51.229 | 71.373 |

| Metric | D0 mean | D3 mean | D3 − D0 |
|---|---:|---:|---:|
| Global R@1 | 5.012 | 8.217 | +3.205 pp |
| Global R@5 | 18.964 | 26.446 | +7.482 pp |
| Global R@10 | 30.133 | 39.916 | +9.783 pp |
| Global R@50 | 65.711 | 73.615 | +7.904 pp |
| Subset R@1 | 32.217 | 28.976 | −3.241 pp |
| Subset R@2 | 54.771 | 52.109 | −2.662 pp |
| Subset R@3 | 73.638 | 71.602 | −2.036 pp |

Thus the stable local pattern is higher global recall but lower CIRR
candidate-set recall for D3.

## 7. LaSCo-inclusive D3-full control

This is a local reconstructed ViT-B, residual-fusion, no-Text-CC experiment
with `CIRR + CIRR-R + LaSCo + Hotels-50K`. The three seeds use the matched
epoch protocol.

| Seed | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.81 | 34.70 | 48.55 | 80.10 | 33.80 | 57.12 | 75.03 |
| 1 | 11.38 | 35.33 | 50.20 | 81.75 | 34.49 | 58.86 | 76.97 |
| 2 | 11.58 | 35.69 | 50.39 | 82.16 | 35.28 | 59.05 | 76.27 |
| **Mean ± sample SD** | **11.26 ± 0.40** | **35.24 ± 0.50** | **49.72 ± 1.01** | **81.34 ± 1.09** | **34.52 ± 0.74** | **58.34 ± 1.07** | **76.09 ± 0.98** |

The strict matched-step seed-0 control reached:

| Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---:|---:|---:|---:|---:|---:|---:|
| 11.72 | 36.40 | 50.01 | 81.66 | 34.75 | 57.52 | 76.54 |

The matched-step result is single-seed evidence. It supports a positive local
LaSCo effect under this ViT-B/no-Text-CC reconstruction, but not the paper's
full ViT-L/ViT-H Text-CC recipe.

## 8. Text-CC scans and implementation evidence

### Multi-source final scan

| Source | Requested | Usable | Raw CC mean | Main loss mean | Raw weighted CC / main loss | Nonzero at `epsilon=0.05` | Nonzero at `epsilon=0.08` |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIRR | 2,000 | 1,955 | 1.0430e−4 | 0.70335 | 1.1871e−5 | 0 | 0 |
| CIRR-R | 1,000 | 1,000 | 1.1128e−4 | 0.70135 | 1.2700e−5 | 0 | 0 |
| LaSCo | 1,000 | 935 | 1.0064e−4 | 0.70636 | 1.1411e−5 | 0 | 0 |
| Hotels | 1,000 | 995 | 1.1266e−4 | 0.69975 | 1.2880e−5 | 0 | 0 |

### Earlier and trained-checkpoint probes

- One cached sample: raw CC `0.0002458533` at `epsilon=0`; thresholded CC
  `0.0000000000` at `epsilon=0.05`.
- CIRR 64-sample scan: 63 usable samples; 63/63 zeroed at `epsilon=0.05`;
  weighted raw CC/main-loss ratio `1.31316e−5`.
- Final CIRR 128-sample scan: 123 usable samples; zero nonzero samples at
  `epsilon=0.08`; mean raw weighted ratio `1.26445e−5`; maximum raw weighted
  ratio `3.76983e−5`.
- LaSCo 128-sample scan: 115 usable samples; zero nonzero samples at both
  `epsilon=0.05` and `0.08`; raw weighted ratio `1.0786e−5`.
- Trained-checkpoint threshold sweep: all tested CIRR-R and Hotels batches had
  exactly zero thresholded CC already at `epsilon=0.005`.

The local residual-fusion synthetic profile also showed much larger early
`q_proj` gradients than the no-residual path. This is an implementation
diagnostic, not proof of a failed paper training run.

## 9. What the evidence supports

1. The reported paper headline was not reproduced under the local protocols.
2. The local D0/D3 control shows a reproducible global-versus-subset trade-off:
   global recall increases while CIRR candidate-set recall decreases.
3. Adding LaSCo improves the locally reconstructed ViT-B/no-Text-CC result,
   including under an approximately matched step budget.
4. The default local Text-CC threshold was inactive on all tested samples in
   the final four-source scan; the unthresholded signal was small relative to
   the main contrastive loss.
5. The exact author-provenance ViT-L/ViT-H Aggregated + Text-CC recipe was not
   reproduced, and the local ViT-H result is a no-NP approximation.

## 10. Evidence index

| Evidence | Source artifact |
|---|---|
| Full historical result narrative | `reproduction_evidence/thesis_reproduction_summary.md` |
| Source/data audit | `reproduction_evidence/final_metrics_report.md` |
| D0/D1/D2/D3 matrix | `public_audit_packet/evidence/reports/session_b_no_np_matrix_result.md` |
| Three-seed D0/D3 confirmation | `reproduction_evidence/session_c_followup_report.md` |
| D0/D3 server results | `public_audit_packet/OFFICIAL_TEST_RESULT_SUMMARY.md` and `evidence/official_test/server_results.csv` |
| LaSCo seed and matched-step result | `public_audit_packet/LASCO_SEED0_RESULT.md` |
| LaSCo Text-CC and seeds 1/2 | `public_audit_packet/LASCO_TEXT_CC_RESULT.md` |
| Text-CC mechanism evidence | `public_audit_packet/MECHANISM_TEXT_CC.md` |
| Final four-source scan | `reproduction_evidence/final_stratified_*_summary.json` |
| Raw submission JSONs and validators | `cirr_test_results/session_b_no_np/` |

