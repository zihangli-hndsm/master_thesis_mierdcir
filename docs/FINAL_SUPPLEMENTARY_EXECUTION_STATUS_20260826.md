# Final supplementary execution status

**Last local audit:** 2026-08-26  
**Scope:** `FINAL_SUPPLEMENTARY_EXPERIMENT_PLAN.md` and `VIT_SCALE_EXPLORATION_CODEX_PLAN.md`

## Status by task

| Item | Status | Evidence / boundary |
|---|---|---|
| Phase 0 artifact/version audit | Complete | Pair-control, checkpoint, source, data, and environment manifests; final release manifest has no missing entries. |
| Task A MiERDCIR package | Complete locally | Three-seed Raw/Fixed/Multi results, hidden-test artifacts, query-style matrix, Surface-50K control, and v2 claim documents. |
| Task B query-style matrix | Complete locally | 27 frozen checkpoint/style evaluations in `narrative_followup/metrics/query_style_matrix.csv`; same 10K pair IDs and text-only style change. |
| Task C rank-delta materials | Automated portion complete; human annotation skipped | CIRR per-query rank deltas, global/subset correlation and text-proxy analysis, 600 candidate rows, 100-row balanced overlap sheet, four contact-sheet PDFs, and a separate caption-only preannotation handoff. Human labels and agreement were intentionally skipped because annotators were unavailable; automatic proxies are not used as labels. |
| Task D intent coverage | Automatic portion complete; human validation skipped | Four-source rule-based coverage, 50K rewrite-output quality audit, pairwise benchmark JS divergence, frequency–performance coefficients, and 24-cell descriptive CIRR performance table. The 400-row manual review sheet is retained blank and no human-validated intent result is reported. |
| Task E ConText-CIR v2 | Complete locally | Complete result consolidation, LaSCo 3-seed mean/std, strict matched-step seed-0 result, corrected Text-CC scan, provenance, source snapshot, and bounded claims. |
| Task F Text-CC scan | Complete locally | CIRR 1,955 usable, CIRR-R 1,000, LaSCo 935, Hotels 995; default threshold nonzero rate is zero in each tested source. |
| Task G D3-full hidden test | Local export complete | Three frozen D3-full checkpoints produced six `rc2` test1 JSON files; all six pass `audit/cirr_validate.py`. CIRR server upload/results are not present. |
| Gate H ViT-L/H pilot | Preparation only | Exact B/L/H model records, configs, manifests, scripts, and gates are prepared; no 2-GPU DDP audit, H200 round-trip, cluster job, or ViT-H result. |
| Gate I Surface-Diverse control | Complete locally | Fixed/Surface/Multi matched 50K control over three seeds; therefore claims remain `intent-routed rewriting` rather than semantic-intent causality. |

## Final package

`docs/CONTEXT_CIR_COMPLETE_RESULTS.md` contains the complete consolidated
ConText-CIR result record, while `docs/CONTEXT_CIR_PAPER_WRITING_INFO.md`
contains paper-ready tables and text. `paper_packages/final_release_20260826/`
contains the current reports,
metrics, official historical artifacts, locally validated D3-full exports,
annotation materials, claims boundary, executable source snapshot, and SHA256
checksums. Large checkpoints are recorded as `external_not_bundled` rather than
copied.

## Explicitly skipped in this release

Human mechanism annotation, double-annotation agreement, and manual intent
validation were intentionally skipped because annotators were unavailable.
The corresponding blank sheets remain in the package for possible future use;
they do not block the automatic evidence package and no human-label,
category-specific, or annotation-based causal claim is made.

## Optional future actions

1. If desired, upload the D3-full JSON files to the CIRR server and append
   returned metrics; local validation is not a server result.
2. Provide the ScienceCluster repository/environment/Slurm values in
   `reproduction_experiments/vit_scale_exploration/INPUTS_REQUIRED.md` before
   freezing or submitting any ViT-L/H job.
