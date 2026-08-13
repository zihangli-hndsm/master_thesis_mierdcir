# Audit Artifacts Index

All artifacts from the 2026-08-13 audit session (plan `NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md`).

## Reports (start here)

| report | content |
|---|---|
| `MTCIR_AUDIT_REPORT.md` | 2×2 cross experiment (7.62 vs 60.62 explanation), batch invariance, leakage, mapping, unit tests, text diagnostics |
| `CIRR_REHEARSAL_REPORT.md` | exporter↔evaluator consistency, batch/order invariance, test1 export smoke, frozen manifest |
| `TEXT_CC_PROBE_REPORT.md` | matched-gradient probe (untrained + converged), ρ_g / cosine / ε threshold |
| `THESIS_METHODS_DRAFT.md` | draft thesis methods: evaluation protocol + caveats |

## Scripts

| script | purpose |
|---|---|
| `audit_mtcir_data.py` | leakage stats, mapping checks, metric unit tests (CPU) |
| `run_mtcir_gpu_audit.sh` | ChromaDB 2×2 + batch invariance (GPU) |
| `run_mtcir_exact_audit.sh` | exact-evaluator 2×2 + batch invariance (GPU) |
| `parse_audit_logs.py` | parse audit logs into tables |
| `run_text_diag.py` | shuffled-text / reference-only diagnostics |
| `cirr_validate.py` | CIRR test1 submission validator |
| `cirr_order_invariance.py` | query-order invariance check |
| `run_cirr_val_rehearsal.sh` | CIRR val rehearsal (export + recompute) |
| `freeze_checkpoints.py` | build frozen checkpoint manifest (9 ckpts + SHA256) |
| `verify_mapping_sample.py` | 100-query mapping spot check |
| `probe_vitl_batch.py` | ViT-L max-batch probe |
| `smoke_vitl_step.py` | real-step memory smoke test |
| `text_cc_gradient_probe.py` | Text CC gradient probe (in audit/ + reproduction_evidence/) |
| `extend_np_cache.py` | NP cache extension for mid/late stages |
| `run_vitl_pilot.sh` | Fixed-L s42 training (started 2026-08-13, cross-session) |
| `run_vitl_remaining.sh` | Multi-L s42 + ViT-B batch-96 matched re-runs (next session) |
| `run_exact_reeval.sh` | exact-evaluator re-eval of the 9 ViT-B checkpoints (final thesis numbers) |

## Data products

| file | content |
|---|---|
| `mtcir_data_audit.json` | leakage + mapping + unit-test numbers |
| `mtcir_gpu_audit_table.json` | ChromaDB 2×2 + batch-invariance table |
| `mapping_sample_100.json` / `mapping_spotcheck_table.md` | manual-check sample (100/100 consistent) |
| `text_diag_0.json` | shuffled/reference-only diagnostics |
| `cirr_pred_metrics.json` | val recompute from exported JSONs |
| `vitl_batch_probe.json` | ViT-L batch probe results (best 128 simple-loss; 96 real step) |

## Outputs elsewhere

| path | content |
|---|---|
| `cirr_test_results/checkpoint_manifest.json` | 9 frozen checkpoints + SHA256 (committed) |
| `cirr_test_results/SUBMISSION_GUIDE.md` | CIRR server upload steps + templates (committed) |
| `cirr_test_results/val_rehearsal/` | val rehearsal exported JSONs |
| `cirr_test_results/raw_s42_recall{,_subset}.json` | test1 export smoke (validated PASS) |
| `logs/audit_mtcir_*.log`, `logs/audit_mtcir_exact_*.log` | raw audit runs |
| `logs/cirr_val_rehearsal_*.log` | rehearsal log |
| `logs/vitl_fixed_s42.log` | Fixed-L training log (live) |
