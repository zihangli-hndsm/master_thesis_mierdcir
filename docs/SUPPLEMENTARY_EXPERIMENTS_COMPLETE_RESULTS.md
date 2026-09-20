# Supplementary experiments: complete results

**Compiled:** 2026-08-27  
**Scope:** all completed non-human-annotation supplementary experiments relevant to the MiERDCIR/rewriting claims.

This document is the consolidated result record. It separates the final matched pair-control evidence from exploratory or historically preliminary runs. Human annotation was not performed and therefore no human-label statistic is included here.

## 1. Evidence hierarchy

| Evidence block | Scope | Status | Use in paper |
|---|---|---|---|
| Pair-control Raw/Fixed/Multi | 255,400 train pairs, 10,000 dev, 10,000 test; 3 seeds | Final matched comparison | Primary evidence |
| Query-style matrix | Same 10,000 test pairs; 9 checkpoints × 3 query styles | Final post-hoc diagnostic | Mechanism/style-alignment evidence |
| Surface-Diverse control | 50,000 training pairs; 3 seeds; 6 lexical frames | Exploratory matched control, scale-confounded | Limitation and alternative-explanation evidence |
| Historical intent-diversity run | Earlier single-run/280K protocol | Preliminary, superseded where the pair-control result exists | Background only |

## 2. Final pair-control protocol

The strict intersection contains 275,400 triplets: 255,400 train, 10,000 dev, and 10,000 test. Raw, Fixed, and Multi use the same pairs, model family (`merdcir_mlp_alpha` / `cross_attn_alpha`), ViT-B backbone, batch size 300, three epochs, and seeds 42, 123, and 2025. Checkpoints are selected only by common-dev mAP on the original MTCIR text; all nine selected checkpoints are epoch 2, step 600.

The split is **pair-held-out, not image-held-out**. It is triplet-disjoint, but 62.9% of test references and 58.6% of test targets also occur in training in another role; 14,018 images overlap across any role. Results must therefore be described as pair-held-out retrieval performance.

### 2.1 Selected checkpoints

| Condition | Seed | Selected step | Common-dev mAP |
|---|---:|---:|---:|
| Raw | 42 | E2 S600 | 0.7172 |
| Raw | 123 | E2 S600 | 0.7164 |
| Raw | 2025 | E2 S600 | 0.7171 |
| Fixed | 42 | E2 S600 | 0.6199 |
| Fixed | 123 | E2 S600 | 0.6189 |
| Fixed | 2025 | E2 S600 | 0.6150 |
| Multi | 42 | E2 S600 | 0.6920 |
| Multi | 123 | E2 S600 | 0.6897 |
| Multi | 2025 | E2 S600 | 0.6847 |

### 2.2 Main evaluation results

Values are mean ± sample SD over the three seeds.

#### MTCIR pair-held-out test

| Training text | Recall@1 | mAP |
|---|---:|---:|
| Raw | **0.6063 ± 0.0009** | **0.7146 ± 0.0006** |
| Multi | 0.5691 ± 0.0046 | 0.6839 ± 0.0029 |
| Fixed | 0.4943 ± 0.0037 | 0.6109 ± 0.0023 |

#### MerdCIR evaluation

| Training text | Recall@1 | mAP |
|---|---:|---:|
| Multi | **0.6788 ± 0.0041** | **0.7748 ± 0.0029** |
| Raw | 0.6233 ± 0.0072 | 0.7282 ± 0.0041 |
| Fixed | 0.5869 ± 0.0020 | 0.6876 ± 0.0021 |

#### FashionIQ development-transfer

| Training text | Recall@1 | mAP |
|---|---:|---:|
| Fixed | **0.0586 ± 0.0013** | **0.0999 ± 0.0014** |
| Multi | 0.0530 ± 0.0007 | 0.0968 ± 0.0011 |
| Raw | 0.0516 ± 0.0018 | 0.0928 ± 0.0021 |

#### CIRR validation development-transfer (4,181 queries)

| Training text | Global R@1 | Global R@5 | Subset R@1 | Subset R@3 |
|---|---:|---:|---:|---:|
| Multi | **0.2586 ± 0.0016** | **0.5477 ± 0.0029** | 0.6373 ± 0.0039 | 0.9206 ± 0.0037 |
| Fixed | 0.2520 ± 0.0032 | 0.5306 ± 0.0022 | **0.6541 ± 0.0043** | **0.9247 ± 0.0016** |
| Raw | 0.2403 ± 0.0014 | 0.5334 ± 0.0041 | 0.6211 ± 0.0050 | 0.9137 ± 0.0022 |

CIRR test1 has no public target labels, so these are validation results, not official hidden-test scores.

### 2.3 Differences relative to Fixed

Percentage-point differences use the three-seed means.

| Dataset / metric | Raw − Fixed | Multi − Fixed |
|---|---:|---:|
| MTCIR R@1 | +11.19 pp | +7.47 pp |
| MTCIR mAP | +10.40 pp | +7.32 pp |
| MerdCIR R@1 | +3.64 pp | +9.19 pp |
| MerdCIR mAP | +4.08 pp | +8.75 pp |
| FashionIQ R@1 | −0.70 pp | −0.56 pp |
| CIRR global R@1 | −1.17 pp | +0.66 pp |
| CIRR global R@5 | +0.28 pp | +1.71 pp |
| CIRR subset R@1 | −3.29 pp | −1.67 pp |
| CIRR subset R@3 | −1.10 pp | −0.41 pp |

### 2.4 Paired bootstrap intervals

The following are paired per-seed differences with 10,000 bootstrap resamples. With only three seeds, these intervals are descriptive rather than a substitute for a larger-seed inferential analysis.

| Contrast | MTCIR R@1 | MerdCIR R@1 | FashionIQ R@1 | CIRR global R@1 | CIRR subset R@1 |
|---|---:|---:|---:|---:|---:|
| Raw − Fixed | +11.19 [+10.88, +11.44] | +3.64 [+3.16, +4.28] | −0.70 [−0.81, −0.57] | −1.17 [−1.65, −0.81] | −3.29 [−3.66, −2.56] |
| Multi − Fixed | +7.47 [+7.18, +7.63] | +9.19 [+8.84, +9.46] | −0.56 [−0.72, −0.39] | +0.66 [+0.12, +1.00] | −1.67 [−1.99, −1.10] |
| Raw − Multi | +3.73 [+3.29, +4.08] | −5.55 [−5.78, −5.18] | −0.14 [−0.33, −0.00] | −1.83 [−1.91, −1.77] | −1.62 [−1.72, −1.46] |

### 2.5 Text statistics on the full 255,400 training pairs

| Statistic | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Mean words | 25.9 | 14.8 | 17.1 |
| Mean CLIP tokens (maximum) | 32.8 (152) | 17.0 (88) | 19.7 (117) |
| Samples over 77 tokens | 0.61% | 0.0004% | 0.008% |
| Document-mean TTR | 0.801 | 0.920 | 0.913 |
| Noun phrases per document | 8.28 | 3.91 | 4.10 |
| “change” per 1k words | 63.6 | 1.9 | 18.9 |
| “remove” per 1k words | 46.6 | 0.06 | 6.3 |
| “add” per 1k words | 44.7 | 0.19 | 3.5 |
| “make” per 1k words | 0.40 | 0.09 | 11.1 |

## 3. Query-style matrix

This post-hoc test evaluates all nine pair-control checkpoints on exactly the same 10,000 test pairs and gallery. Only the query text changes. Values are mean ± sample SD over seeds 42, 123, and 2025.

### Recall@1

| Training condition \\ query | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Raw | 0.6063 ± 0.0009 | 0.7713 ± 0.0014 | 0.5321 ± 0.0008 |
| Fixed | 0.4943 ± 0.0037 | 0.8254 ± 0.0009 | 0.5082 ± 0.0014 |
| Multi | 0.5691 ± 0.0046 | 0.7888 ± 0.0028 | 0.5891 ± 0.0031 |

### Recall@5

| Training condition \\ query | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Raw | 0.8515 ± 0.0021 | 0.9472 ± 0.0008 | 0.7896 ± 0.0016 |
| Fixed | 0.7544 ± 0.0010 | 0.9621 ± 0.0009 | 0.7416 ± 0.0012 |
| Multi | 0.8300 ± 0.0009 | 0.9533 ± 0.0005 | 0.8354 ± 0.0011 |

### Recall@10

| Training condition \\ query | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Raw | 0.9109 ± 0.0017 | 0.9743 ± 0.0006 | 0.8674 ± 0.0005 |
| Fixed | 0.8332 ± 0.0001 | 0.9818 ± 0.0004 | 0.8139 ± 0.0015 |
| Multi | 0.8972 ± 0.0004 | 0.9790 ± 0.0006 | 0.9023 ± 0.0013 |

### Recall@50

| Training condition \\ query | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Raw | 0.9776 ± 0.0005 | 0.9966 ± 0.0002 | 0.9627 ± 0.0008 |
| Fixed | 0.9442 ± 0.0003 | 0.9967 ± 0.0000 | 0.9266 ± 0.0011 |
| Multi | 0.9734 ± 0.0008 | 0.9967 ± 0.0002 | 0.9771 ± 0.0004 |

### mAP

| Training condition \\ query | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Raw | 0.7146 ± 0.0006 | 0.8488 ± 0.0011 | 0.6478 ± 0.0011 |
| Fixed | 0.6109 ± 0.0023 | 0.8862 ± 0.0003 | 0.6148 ± 0.0016 |
| Multi | 0.6839 ± 0.0029 | 0.8618 ± 0.0017 | 0.6982 ± 0.0015 |

The diagonal is consistently strongest for the matching query style: Raw→Raw, Fixed→Fixed, and Multi→Multi. Fixed queries are also easiest overall. This supports query-style alignment, but does not by itself identify semantic intent diversity as the causal mechanism because lexical surface form also changes.

## 4. Surface-Diverse exploratory control

Surface-Diverse uses six deterministic lexical frames expressing a generic retrieval intent. The final matched analysis uses the same 50,000 training pairs, 10,000 test pairs, and three seeds (42, 123, 2025) for Fixed, Surface, and Multi. It is still exploratory because the training set is 50K rather than 255.4K; the scale difference prevents a clean causal comparison with the main pair-control experiment.

All 50,000 rewrites and NP extractions succeeded. Frame proportions were 15.8–16.7%, and the generated outputs retained their assigned frame. Mean length was 24.6 words, compared with 14.8 for Fixed and 17.1 for Multi.

### MTCIR test matrix (mean ± sample SD; rows are training condition, columns are query style)

| Metric | Train | Raw | Fixed | Multi | Surface |
|---|---|---:|---:|---:|---:|
| R@1 | Fixed | 0.3115 ± 0.0017 | 0.6079 ± 0.0022 | 0.3248 ± 0.0027 | 0.5642 ± 0.0025 |
| R@1 | Surface | 0.3090 ± 0.0005 | 0.5944 ± 0.0025 | 0.3146 ± 0.0015 | 0.5829 ± 0.0012 |
| R@1 | Multi | 0.2927 ± 0.0020 | 0.5184 ± 0.0044 | 0.3196 ± 0.0048 | 0.4856 ± 0.0040 |
| R@5 | Fixed | 0.5895 ± 0.0068 | 0.8490 ± 0.0017 | 0.5983 ± 0.0044 | 0.8241 ± 0.0020 |
| R@5 | Surface | 0.5861 ± 0.0013 | 0.8398 ± 0.0017 | 0.5823 ± 0.0015 | 0.8337 ± 0.0003 |
| R@5 | Multi | 0.5825 ± 0.0008 | 0.7940 ± 0.0014 | 0.6129 ± 0.0037 | 0.7728 ± 0.0023 |
| R@10 | Fixed | 0.7004 ± 0.0036 | 0.9124 ± 0.0007 | 0.7032 ± 0.0029 | 0.8906 ± 0.0014 |
| R@10 | Surface | 0.6980 ± 0.0011 | 0.9043 ± 0.0010 | 0.6884 ± 0.0039 | 0.8999 ± 0.0015 |
| R@10 | Multi | 0.6982 ± 0.0023 | 0.8728 ± 0.0008 | 0.7242 ± 0.0018 | 0.8564 ± 0.0041 |
| R@50 | Fixed | 0.8939 ± 0.0044 | 0.9809 ± 0.0001 | 0.8858 ± 0.0015 | 0.9748 ± 0.0010 |
| R@50 | Surface | 0.8913 ± 0.0024 | 0.9788 ± 0.0006 | 0.8744 ± 0.0021 | 0.9776 ± 0.0006 |
| R@50 | Multi | 0.8963 ± 0.0025 | 0.9727 ± 0.0006 | 0.9078 ± 0.0010 | 0.9665 ± 0.0007 |
| mAP | Fixed | 0.4416 ± 0.0028 | 0.7157 ± 0.0012 | 0.4523 ± 0.0024 | 0.6798 ± 0.0022 |
| mAP | Surface | 0.4379 ± 0.0004 | 0.7038 ± 0.0019 | 0.4403 ± 0.0015 | 0.6945 ± 0.0006 |
| mAP | Multi | 0.4270 ± 0.0012 | 0.6425 ± 0.0029 | 0.4545 ± 0.0042 | 0.6141 ± 0.0030 |

### CIRR validation per seed

| Train | Seed | Metric | R@1 | R@5 / R@2 | R@10 / R@3 | mAP |
|---|---:|---|---:|---:|---:|---:|
| Fixed | 123 | global | 0.1834 | 0.4269 | 0.5582 | 0.3006 |
| Fixed | 123 | subset | 0.5348 | 0.7458 | 0.8744 | 0.6832 |
| Fixed | 2025 | global | 0.1729 | 0.4162 | 0.5525 | 0.2887 |
| Fixed | 2025 | subset | 0.5214 | 0.7443 | 0.8701 | 0.6748 |
| Fixed | 42 | global | 0.1791 | 0.4205 | 0.5602 | 0.2970 |
| Fixed | 42 | subset | 0.5276 | 0.7522 | 0.8740 | 0.6805 |
| Multi | 123 | global | 0.1509 | 0.4018 | 0.5346 | 0.2706 |
| Multi | 123 | subset | 0.4884 | 0.7096 | 0.8491 | 0.6455 |
| Multi | 2025 | global | 0.1440 | 0.3891 | 0.5264 | 0.2641 |
| Multi | 2025 | subset | 0.4767 | 0.7056 | 0.8488 | 0.6389 |
| Multi | 42 | global | 0.1478 | 0.4056 | 0.5326 | 0.2692 |
| Multi | 42 | subset | 0.4922 | 0.7127 | 0.8462 | 0.6470 |
| Surface | 123 | global | 0.1693 | 0.4155 | 0.5480 | 0.2869 |
| Surface | 123 | subset | 0.5283 | 0.7426 | 0.8699 | 0.6779 |
| Surface | 2025 | global | 0.1624 | 0.3994 | 0.5350 | 0.2777 |
| Surface | 2025 | subset | 0.5166 | 0.7458 | 0.8646 | 0.6708 |
| Surface | 42 | global | 0.1701 | 0.4152 | 0.5513 | 0.2898 |
| Surface | 42 | subset | 0.5281 | 0.7517 | 0.8711 | 0.6797 |

The low absolute MTCIR result of this 50K experiment is a data-size effect and must not be interpreted as evidence that surface diversity is harmful. The control establishes that the six templates can be generated and retained reliably; it does not establish a full-scale surface-diversity effect.

## 5. Historical preliminary record

An earlier single-run/280K Fixed-vs-Multi experiment reported Fixed/Multi MTCIR R@1 of 0.4217/0.4717 and MerdCIR R@1 of 0.6240/0.7040, with additional transfer metrics in `docs/INTENT_DIVERSITY_REPORT.md`. These values used an earlier protocol and are retained for traceability only. The final three-seed pair-control results above supersede them for claims about the main comparison. In particular, the earlier overfitting interpretation is not confirmed: in the final three-epoch pair-control runs, all three conditions continued to improve through the selected checkpoint.

## 6. Overall result and limitations

1. Raw is strongest on the original MTCIR pair-held-out test, while Multi is strongest on MerdCIR.
2. Multi has a reproducible source-domain advantage over Fixed and a large MerdCIR advantage, but the evidence does not isolate semantic intent diversity from routed rewriting and surface-form differences.
3. No general cross-dataset advantage is established: Fixed is best on FashionIQ and CIRR subset metrics, while Multi has only mixed/global CIRR gains.
4. Reduced overfitting is not established because the three-epoch budget did not show post-peak decline.
5. Surface-Diverse is a useful negative/feasibility control, but its 50K-vs-255K scale confound prevents a causal surface-diversity claim.
6. CIRR and FashionIQ results are development-transfer evaluations; CIRR test1 hidden-test performance is not locally verifiable.
7. The exact gallery evaluator was audited: the old ChromaDB/HNSW path differs from exact matmul by at most 2e-4, and the final conclusions are unchanged.

## 7. Source artifacts

- Primary narrative: `docs/FOLLOWUP_EXPERIMENT_FINAL_REPORT.md`
- Pair-control report: `docs/PAIR_CONTROL_RESULTS.md`
- Surface report: `docs/SURFACE_DIVERSE_REPORT.md`
- Query-style report: `narrative_followup/reports/query_style_matrix_report.md`
- Surface matched-control report: `narrative_followup/reports/surface_control_report.md`
- Pair-control metrics: `checkpoints/pair_control_all_metrics.json`, `checkpoints/pair_control_eval_summary.json`, `checkpoints/pair_control_text_stats.json`, `checkpoints/pair_control_bootstrap_ci.json`
- Query-style metrics: `narrative_followup/metrics/query_style_matrix.csv`, `query_style_matrix_per_seed.csv`, and `query_style_ranks.csv`
- Surface metrics: `narrative_followup/metrics/surface_control_mtcir_matrix.csv` and `surface_control_cirr_metrics.csv`
