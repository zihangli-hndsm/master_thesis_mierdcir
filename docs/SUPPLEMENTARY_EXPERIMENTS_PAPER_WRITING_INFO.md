# Supplementary experiments: paper-writing information

This is the paper-facing extraction from `SUPPLEMENTARY_EXPERIMENTS_COMPLETE_RESULTS.md`. It is intentionally separate from the full audit record.

## Recommended central claim

Use the following conservative claim:

> Multi-intent rewriting improves retrieval on the rewritten MerdCIR domain and provides a reproducible source-domain advantage over the Fixed rewrite under pair-held-out evaluation. However, the experiments do not establish a general cross-dataset transfer benefit or isolate semantic intent diversity from the combined effects of rewriting policy, lexical routing, and query-style alignment.

## Main numbers to cite

| Comparison | Result |
|---|---:|
| MTCIR Raw vs Fixed, R@1 | +11.19 pp (0.6063 vs 0.4943) |
| MTCIR Multi vs Fixed, R@1 | +7.47 pp (0.5691 vs 0.4943) |
| MerdCIR Multi vs Fixed, R@1 | +9.19 pp (0.6788 vs 0.5869) |
| FashionIQ Multi vs Fixed, R@1 | −0.56 pp (0.0530 vs 0.0586) |
| CIRR global Multi vs Fixed, R@1 | +0.66 pp (0.2586 vs 0.2520) |
| CIRR subset Multi vs Fixed, R@1 | −1.67 pp (0.6373 vs 0.6541) |

All main values are mean ± sample SD over three seeds: 42, 123, and 2025.

## Suggested results paragraph

> Under a frozen pair-control split and identical training settings, Raw text achieved the highest MTCIR test performance (R@1 60.63, mAP 71.46), whereas Multi achieved the highest MerdCIR performance (R@1 67.88, mAP 77.48). Relative to Fixed, Multi improved MTCIR R@1 by 7.47 percentage points and MerdCIR R@1 by 9.19 points. This improvement did not transfer uniformly: Multi was slightly below Fixed on FashionIQ R@1 and on CIRR candidate-subset R@1, although it was slightly higher on CIRR global R@1 and R@5. Thus, the results support a domain- and query-style-dependent benefit rather than a universal generalization improvement.

## Suggested mechanism paragraph

> A post-hoc query-style matrix evaluated the same checkpoints and image pairs with Raw, Fixed, or Multi text. The best result for each training condition occurred on the matching style (Raw→Raw, Fixed→Fixed, and Multi→Multi), indicating query-style alignment. Fixed queries were easiest overall. Because the styles differ in both semantic routing and surface realization, this analysis is diagnostic rather than a causal test of semantic intent diversity.

## Suggested limitation paragraph

> The principal comparison is pair-held-out rather than image-held-out: the split prevents repeated triplets but allows images to appear in different roles across train and test. CIRR and FashionIQ results are development-transfer evaluations, and CIRR test1 hidden-test targets are unavailable locally. The Surface-Diverse control used 50K rather than 255.4K training pairs; it validates generation and template retention but cannot identify the causal effect of surface diversity. Human annotation was not completed and no annotation-based claim is made.

## Hypothesis decisions

| Hypothesis | Decision | Safe wording |
|---|---|---|
| H1: rewriting is universally beneficial | Not supported | Raw is strongest on the original MTCIR style; rewriting benefits are domain-dependent. |
| H2: Multi beats Fixed in the source/rewritten domain | Supported for the measured domains | Multi is consistently better than Fixed on MTCIR and MerdCIR in the three-seed pair-control. |
| H3: Multi improves cross-dataset generalization | Not supported | Transfer results are mixed and do not show a uniform advantage. |
| H4: Multi reduces overfitting | Not established | The three-epoch runs did not reach a post-peak regime. |

## Wording to avoid

- “image-held-out generalization” — use “pair-held-out evaluation”.
- “Multi improves generalization” — use “Multi improves MerdCIR/source-domain retrieval; transfer is mixed”.
- “semantic intent diversity is the sole cause” — the Surface control and query-style matrix do not permit that attribution.
- “official CIRR test result” — the reported CIRR values are validation/development-transfer metrics unless separately marked as hidden-test submission results.
- “human study confirms…” — annotation was skipped.

## Tables and figures to place in a thesis

1. Main supplementary table: pair-control MTCIR, MerdCIR, FashionIQ, and CIRR validation results.
2. Mechanism table: the 3×3 query-style R@1/mAP matrix, optionally with R@5/R@10/R@50 in the appendix.
3. Text-statistics table: length, truncation, TTR, NP count, and action-word frequencies.
4. Limitation table: Surface-Diverse full metric matrix, explicitly marked 50K exploratory control.

## Reproducibility statement

The full values, per-seed outputs, checkpoint-selection rule, split caveat, evaluator audit, and artifact paths are recorded in `SUPPLEMENTARY_EXPERIMENTS_COMPLETE_RESULTS.md`. The current release package also contains the raw metric files and source reports; no human annotation artifact is presented as a completed label set.
