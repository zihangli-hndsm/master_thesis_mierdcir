# Follow-up Experiments on Text Rewriting and Intent Diversity in MiERDCIR

**Status:** Core follow-up experiment complete; ECIR-oriented report draft  
**Date:** 2026-08-18  
**Project:** ScheiCIR / MiERDCIR thesis follow-up study

## Abstract

This report presents a controlled follow-up study of text rewriting for composed image retrieval. The study was motivated by an earlier result suggesting that VLM-rewritten modification text and multi-intent rewriting substantially improve retrieval performance over the original MTCIR supervision. We construct a frozen pair-controlled corpus and compare three conditions using the same image pairs, model architecture, optimization settings, checkpoint-selection rule, and three training seeds: Raw, which uses the original MTCIR modification text; Fixed, which uses a single naturalization prompt; and Multi, which uses six intent-routed rewriting prompts.

The core experiment is complete. Each condition was trained with seeds 42, 123, and 2025. Checkpoints were selected only on a frozen common-development set, and the final CIRR test predictions were submitted before the server results were recorded. The results do not support a single universal claim that rewriting improves retrieval. Raw is strongest on the source-domain held-out MTCIR test, while Multi is clearly stronger than Fixed on MTCIR and MerdCIR. On CIRR, Multi improves full-gallery recall relative to Fixed, but is worse on candidate-set recall. FashionIQ shows no meaningful Multi advantage.

The resulting evidence supports a more specific conclusion: multi-intent rewriting changes the learned text supervision in a reproducible and source-domain-beneficial way, but its independent cross-dataset generalization benefit is not established. The study also identifies an important validity boundary: the pair-controlled split is triplet-disjoint but not image-disjoint, so the source-domain result must be described as pair-held-out rather than image-held-out generalization. The report therefore frames the contribution as a controlled negative/qualified result about when rewriting helps, rather than as evidence for unconditional rewriting superiority.

## 1. Motivation and research questions

The earlier thesis experiments compared models trained with original MTCIR text and VLM-rewritten text, but several alternative explanations remained possible:

- the image pairs might differ across conditions;
- the rewritten corpus might change the effective training set;
- a single fixed prompt might introduce a surface-form artifact;
- checkpoint selection might use a benchmark that was later reported as transfer performance;
- a single seed might explain the observed improvement;
- the previous low MTCIR result might reflect an old weak checkpoint rather than the text formulation.

The follow-up study was designed to answer four questions.

**RQ1.** Does rewriting improve retrieval when image pairs, training budget, model, and checkpoint protocol are held fixed?

**RQ2.** Does six-way intent routing provide an independent benefit over a single naturalization prompt?

**RQ3.** Does any benefit transfer to CIRR and FashionIQ, including both full-gallery and candidate-set CIRR retrieval?

**RQ4.** Are the observed differences stable across training seeds and robust to the hidden CIRR test evaluation?

## 2. Experimental conditions

### 2.1 Data and pair control

The three conditions use the same frozen strict-intersection universe of 275,400 triplets:

- 255,400 training pairs;
- 10,000 common-development pairs;
- 10,000 source-domain held-out test pairs.

The data manifest is `data/pair_control/manifest.json`. The three conditions differ only in the text field used for the same reference-target pairs:

| Condition | Training text | Purpose |
|---|---|---|
| Raw | Original MTCIR modification text | Original supervision baseline |
| Fixed | One naturalization/rewrite prompt | Overall rewriting and surface-naturalization control |
| Multi | Six intent-routed rewriting prompts | Intent-diversity condition |

### 2.2 Model and optimization

All core runs use the same configuration:

- `merdcir_mlp_alpha` training path (`cross_attn_alpha` model);
- CLIP ViT-B/32 backbone;
- batch size 300;
- three epochs;
- same optimizer, learning rate, image preprocessing, and maximum training protocol;
- Text-CC/soft contrastive loss disabled for this thesis follow-up (`sc_lambda=0`);
- seeds 42, 123, and 2025 for each condition.

This design isolates the text-supervision variable. It does not test Text-CC; that mechanism belongs to the separate ConText-CIR audit.

### 2.3 Checkpoint selection

For every condition and seed, checkpoints were evaluated every 300 steps on the frozen common-development set using the original MTCIR text. The selected checkpoint was:

\[
c^* = \arg\max_c \operatorname{mAP}_{\mathrm{common-dev}}(c).
\]

All nine runs selected the epoch-2, step-600 checkpoint. CIRR test and FashionIQ results were not used for checkpoint selection. The frozen checkpoint manifest is [checkpoint_manifest.json](../cirr_test_results/checkpoint_manifest.json).

### 2.4 Evaluation sets

The study reports four evaluation categories:

1. **MTCIR source-domain held-out test:** 10,000 frozen pair-held-out queries.
2. **MerdCIR evaluation:** evaluation with rewritten text.
3. **CIRR:** local validation for development-transfer analysis and official hidden `test1` server results.
4. **FashionIQ:** development-transfer evaluation.

The CIRR test server outputs are in [server_results.csv](../cirr_test_results/server_results.csv), and all 18 submitted prediction files have matching validation files with `all_checks_passed: true`.

## 3. Integrity and protocol checks

The following checks were completed before interpreting the final results:

- all three conditions use the same pair-controlled ID universe;
- all nine checkpoint choices follow the common-dev rule;
- no CIRR test result was used for model selection;
- 18 official CIRR submissions were generated: 3 conditions × 3 seeds × 2 metrics;
- all submission hashes match `submission_manifest.json`;
- all 18 submission validators pass;
- exact evaluator and batch-size checks show that the local global-gallery evaluator is batch-invariant up to negligible ANN noise;
- the old MTCIR R@1=7.62% result was reproduced with the old checkpoint and old split, while the new pair-control checkpoint reaches approximately 60.6% on the new held-out split. The discrepancy is therefore primarily checkpoint/training quality, not an evaluator bug.

The final integrity check found no missing manifest entries, hash mismatch, or invalid submission. The exact evaluator audit and old/new checkpoint cross experiment are documented in [MTCIR_AUDIT_REPORT.md](../audit/MTCIR_AUDIT_REPORT.md).

The source-domain MTCIR and MerdCIR numbers below use the exact-matmul reevaluation from `checkpoints/pair_control_exact_metrics_20260816.json`. Earlier ChromaDB values differ only by negligible ANN noise and do not change any conclusion.

## 4. Results

### 4.1 Source-domain held-out MTCIR test

Mean ± sample standard deviation over three seeds:

| Condition | Recall@1 | mAP |
|---|---:|---:|
| Raw | **0.6063 ± 0.0009** | **0.7146 ± 0.0006** |
| Multi | 0.5691 ± 0.0046 | 0.6839 ± 0.0029 |
| Fixed | 0.4943 ± 0.0037 | 0.6109 ± 0.0023 |

Raw is strongest on the original-text source-domain test. Fixed is substantially worse than Raw, while Multi recovers part of the loss and is clearly stronger than Fixed.

Relative to Fixed, the mean Recall@1 differences are:

- Raw: `+11.20` percentage points;
- Multi: `+7.48` percentage points;
- Raw versus Multi: `+3.72` percentage points.

The result does not support the claim that rewriting is universally better than the original text. It does support an independent Multi-over-Fixed source-domain effect.

### 4.2 MerdCIR evaluation

| Condition | Recall@1 | mAP |
|---|---:|---:|
| Multi | **0.6788 ± 0.0041** | **0.7748 ± 0.0029** |
| Raw | 0.6233 ± 0.0072 | 0.7282 ± 0.0041 |
| Fixed | 0.5869 ± 0.0020 | 0.6876 ± 0.0021 |

Multi is strongest on the rewritten-text evaluation set. Relative to Fixed, the Multi advantage is `+9.19` percentage points in Recall@1 and `+8.73` percentage points in mAP. This is the clearest evidence that intent-routed rewriting changes the model in a way that is beneficial for the rewritten MerdCIR distribution.

However, this is not a fully independent natural-language transfer test because the Multi training distribution and MerdCIR evaluation distribution are related. It should be described as source/rewritten-domain performance, not universal generalization.

### 4.3 FashionIQ transfer

| Condition | Recall@1 | mAP |
|---|---:|---:|
| Fixed | **0.0586 ± 0.0013** | **0.0999 ± 0.0014** |
| Multi | 0.0530 ± 0.0007 | 0.0968 ± 0.0011 |
| Raw | 0.0516 ± 0.0018 | 0.0928 ± 0.0021 |

Multi does not improve FashionIQ over Fixed. The differences are small in absolute terms and do not support a claim that six-way intent diversity improves transfer to a different CIR task.

### 4.4 Local CIRR validation

| Condition | Global R@1 | Global R@5 | Subset R@1 | Subset R@3 |
|---|---:|---:|---:|---:|
| Multi | **0.2586 ± 0.0016** | **0.5477 ± 0.0029** | 0.6373 ± 0.0039 | 0.9206 ± 0.0037 |
| Fixed | 0.2520 ± 0.0032 | 0.5306 ± 0.0022 | **0.6541 ± 0.0043** | **0.9247 ± 0.0016** |
| Raw | 0.2403 ± 0.0014 | 0.5334 ± 0.0041 | 0.6211 ± 0.0050 | 0.9137 ± 0.0022 |

Multi improves global CIRR ranking relative to Fixed, but Fixed remains better on candidate-set retrieval. This distinction is central: a full-gallery improvement is not equivalent to a uniform improvement in compositional retrieval quality.

### 4.5 Official CIRR test server

The official test results are mean ± sample standard deviation over seeds 42, 123, and 2025:

| Condition | Global R@1 | Global R@5 | Global R@10 | Global R@50 | Subset R@1 | Subset R@2 | Subset R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 25.542 ± 0.230 | 53.887 ± 0.451 | 66.506 ± 0.313 | 88.490 ± 0.279 | 62.618 ± 0.471 | 82.056 ± 0.216 | 91.124 ± 0.074 |
| Fixed | 26.594 ± 0.458 | 54.506 ± 0.377 | 66.595 ± 0.264 | 87.526 ± 0.113 | **64.900 ± 0.174** | **83.357 ± 0.217** | **92.362 ± 0.245** |
| Multi | **26.932 ± 0.202** | **56.000 ± 0.515** | **68.803 ± 0.282** | **89.397 ± 0.272** | 63.863 ± 0.171 | 82.980 ± 0.196 | 91.590 ± 0.261 |

The hidden test confirms the validation pattern:

- Multi is best on all reported global Recall metrics;
- Fixed is best on all three subset metrics;
- Multi versus Fixed is `+0.337` pp on global R@1, `+1.494` pp on R@5, `+2.209` pp on R@10, and `+1.871` pp on R@50;
- Multi versus Fixed is `−1.036` pp on subset R@1, `−0.377` pp on subset R@2, and `−0.771` pp on subset R@3.

Thus, the final test result is not a generic win for Multi. It indicates a metric- and retrieval-setting-dependent effect: Multi improves full-gallery ranking but does not improve the restricted candidate-set task.

## 5. Text-property evidence

The full training corpus statistics are:

| Statistic | Raw | Fixed | Multi |
|---|---:|---:|---:|
| Mean words | 25.9 | 14.8 | 17.1 |
| Mean CLIP tokens | 32.8 | 17.0 | 19.7 |
| Over 77 CLIP tokens | 0.61% | 0.0004% | 0.008% |
| Type-token ratio | 0.801 | 0.920 | 0.913 |
| Noun phrases/document | 8.28 | 3.91 | 4.10 |

Rewriting substantially shortens the text and nearly eliminates CLIP-token truncation. Fixed and Multi are much closer to each other in length, TTR, and noun-phrase count than either is to Raw. Their main controlled difference is the intent-routing policy and the associated action-verb distribution.

This supports the interpretation that Multi-over-Fixed differences are not simply caused by text length or truncation. It does not, however, completely isolate semantic intent diversity from surface-form diversity. A full-size Surface-Diverse control remains desirable for that stronger claim.

## 6. Hypothesis decisions

### H1: Rewriting improves retrieval overall

**Not supported as a universal claim.** Raw is substantially better than Fixed and Multi on the source-domain MTCIR test. Rewriting is therefore not uniformly beneficial under pair-controlled conditions.

A qualified statement is supported:

> Rewriting changes the learned retrieval behavior, but its benefit depends on the evaluation distribution. A single naturalization prompt is harmful on the original-text source test, whereas multi-intent rewriting recovers performance and improves the rewritten-text distribution.

### H2: Intent diversity provides an independent contribution

**Supported for source/rewritten-domain retrieval, with a remaining surface-diversity caveat.** Multi beats Fixed across all three seeds on MTCIR and MerdCIR. The effect is large and consistent:

- MTCIR test Recall@1: `+7.47` pp;
- MerdCIR Recall@1: `+9.19` pp;
- MerdCIR mAP: `+8.75` pp.

The evidence is sufficient to report a reproducible Multi-over-Fixed effect, but the most precise attribution is “six-way intent-routed rewriting” rather than “semantic intent diversity alone,” because the formal full-size Surface-Diverse control was not completed.

### H3: Intent diversity improves cross-dataset generalization

**Not supported.** Multi has no advantage over Fixed on FashionIQ and loses on CIRR subset retrieval. It improves CIRR global recall, including the official hidden test, but this does not extend to all CIRR metrics.

Recommended wording:

> Multi-intent rewriting improves source/rewritten-domain retrieval and full-gallery CIRR ranking, while its independent benefit for cross-dataset or candidate-set generalization is not established.

### H4: Intent diversity reduces overfitting

**Not established.** The main three-epoch runs do not show a post-peak decline under the common-dev protocol. The earlier exploratory curve suggested a difference in convergence behavior, but it is not sufficient to claim reduced overfitting without a longer matched-budget learning-curve analysis.

## 7. Main interpretation

The experiments support a distributional interpretation rather than a universal “better text” interpretation.

Raw retains the original MTCIR language distribution and is therefore strongest on the original-text source test. Fixed makes the text shorter and more natural but removes much of the original modification vocabulary and structure; this harms source-domain performance. Multi introduces a richer set of rewrite behaviors and recovers source-domain performance relative to Fixed. It also performs best on MerdCIR, whose text distribution is related to the multi-intent rewriting process.

On CIRR, Multi improves full-gallery ranking but not candidate-set ranking. This may reflect a change in global image prior or broad ranking behavior rather than a uniform improvement in fine-grained reference-conditioned composition. This is why the report does not use the phrase “improves generalization” without qualification.

## 8. Threats to validity and required wording

### 8.1 Pair-held-out, not image-held-out

The pair-control split has zero triplet overlap, but it is not image-disjoint. The audit found that approximately 62.9% of test references and 58.6% of test targets also occur in training in another triplet role, with 14,018 images overlapping in at least one role. Consequently:

- MTCIR source-test results are pair-held-out retrieval results;
- they must not be described as image-held-out generalization;
- an image-disjoint split is required for that stronger claim.

### 8.2 Surface-Diverse control

A 50K, one-seed Surface-Diverse exploration confirmed that six surface templates can be generated and preserved, but its lower score is confounded by the 50K versus 255K training size. It is not a formal ablation. The final report therefore uses “intent-routed rewriting” and not the stronger isolated causal phrase “semantic intent diversity alone.”

### 8.3 Development-transfer status

FashionIQ and local CIRR validation were used as development-transfer evaluations during the study. The official CIRR hidden test is the strongest transfer evidence in this report. Future experiments should freeze all remaining choices before using additional hidden test submissions.

### 8.4 Limited number of seeds

Three paired training seeds satisfy the minimum planned replication level and show highly stable directionality, but they are not a substitute for a large-sample statistical analysis. The report therefore gives per-seed values and mean ± standard deviation, while treating bootstrap intervals as descriptive.

### 8.5 Dataset scale

The completed controlled matrix uses the frozen 255,400-pair training split, not the full 3.4M-corpus scale described in the earlier plan. Scaling to 3.4M is a separate experiment and cannot be inferred from the present results.

## 9. Final conclusion

The core follow-up experiment is complete and reproducible at the controlled 255,400-pair scale. The strongest defensible conclusions are:

1. A single naturalization rewrite is not universally beneficial; it substantially lowers performance on the original-text source-domain test.
2. Six-way intent-routed rewriting consistently outperforms the fixed rewrite condition on MTCIR and MerdCIR across three seeds.
3. Multi-intent rewriting improves full-gallery CIRR recall on the official hidden test, but Fixed remains better on all CIRR candidate-set metrics.
4. The evidence does not establish a general cross-dataset or image-held-out generalization benefit.
5. The main scientific contribution of this follow-up is a controlled decomposition showing that text rewriting, intent routing, source-domain alignment, and transfer behavior are not interchangeable claims.

For an ECIR-oriented submission, the paper should present this as a careful controlled study of rewriting and intent diversity, with explicit negative results and validity boundaries. The current evidence is sufficient for the core experimental report. It is not sufficient to claim image-disjoint generalization, full 3.4M scaling, or a pure causal effect of semantic intent diversity without the remaining controls.

## 10. Reproducibility artifacts

- Experiment protocol: [FOLLOWUP_EXPERIMENT_PLAN.md](FOLLOWUP_EXPERIMENT_PLAN.md)
- Core result summary: [PAIR_CONTROL_RESULTS.md](PAIR_CONTROL_RESULTS.md)
- Official CIRR results: [server_results.csv](../cirr_test_results/server_results.csv)
- Submission manifest: [submission_manifest.json](../cirr_test_results/submission_manifest.json)
- Frozen checkpoint manifest: [checkpoint_manifest.json](../cirr_test_results/checkpoint_manifest.json)
- MTCIR evaluator audit: [MTCIR_AUDIT_REPORT.md](../audit/MTCIR_AUDIT_REPORT.md)
- Surface-Diverse exploratory analysis: [SURFACE_DIVERSE_REPORT.md](SURFACE_DIVERSE_REPORT.md)
- Pair-control metrics: `checkpoints/pair_control_all_metrics.json`
- Bootstrap summaries: `checkpoints/pair_control_bootstrap_ci.json`

Before public release, add the exact environment lockfile, the final source diff/commit, data-manifest hashes, and the selected checkpoint hashes to the artifact package.
