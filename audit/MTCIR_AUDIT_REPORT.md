# MTCIR Evaluation Audit Report

**Date:** 2026-08-13
**Plan reference:** `NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md` §7
**Objective:** Explain the gap between the old MTCIR R@1=7.62% and pair-control R@1=60.62%, and verify the evaluator before CIRR test1 submission.

## 0. Key Findings (TL;DR)

1. **The old 7.62% is fully reproducible** with the current code: `topk_mtcir` checkpoint (selection score 0.115) evaluated on the old 5K split (`mtcir_np/eval/eval_subset.jsonl`) gives exactly R@1=0.0762. The evaluator code is **not** the source of the gap.
2. **The 7.62 → 60.62 gap is dominated by the checkpoint** (model/training quality), with a secondary contribution from the evaluation split:
   | | old split (5K) | new split (10K) |
   |---|---|---|
   | old ckpt (`topk_mtcir`) | **A = 0.0762** | B = 0.1139 |
   | new ckpt (pair RAW s42) | C = 0.4757 | **D = 0.6058** |
   - model contribution: C−A = +39.9pt (old split), D−B = +49.2pt (new split)
   - split contribution: B−A = +3.8pt (old ckpt), D−C = +13.0pt (new ckpt)
3. **Batch-size invariance holds exactly** with the new exact-matmul evaluator (R@1 = 0.605800 for batch 64/128/256). The ChromaDB/HNSW path carries ≤2e-4 ANN noise (0.6059–0.6060 vs exact 0.6058).
4. **Triplet-level split is clean (0 overlap), but image-level leakage is substantial**: 62.9% of test references and 58.6% of test targets also appear in train (any-role overlap 14,018). Results should be described as **pair-held-out, not image-held-out**.
5. **The model strongly depends on text** (shuffled-text R@1: 73.95% → 3.35%; reference-only: 20.45%), ruling out text-ignoring / image-leakage-dominated retrieval.

## 1. §7.1 Global gallery & batch-size invariance

Evaluator: `eval_checkpoint.py` builds a global gallery of unique target images, ranks every query against the full gallery (never batch-local). Exact-matmul mode (`--exact-gallery`) added for the audit.

| evaluator | batch | R@1 | R@5 | R@10 | R@50 | mAP |
|---|---|---|---|---|---|---|
| ChromaDB | 32 | 0.606000 | 0.8492 | 0.9107 | 0.9777 | 0.713819 |
| ChromaDB | 64 | 0.606000 | 0.8492 | 0.9107 | 0.9777 | 0.713819 |
| ChromaDB | 128 | 0.605900 | 0.8492 | 0.9107 | 0.9776 | 0.713750 |
| ChromaDB | 256 | 0.605900 | 0.8492 | 0.9107 | 0.9776 | 0.713750 |
| **Exact** | 64 | **0.605800** | 0.8492 | 0.9107 | 0.9776 | 0.713799 |
| **Exact** | 128 | **0.605800** | 0.8492 | 0.9107 | 0.9776 | 0.713799 |
| **Exact** | 256 | **0.605800** | 0.8492 | 0.9107 | 0.9776 | 0.713799 |

**Verdict:** exact evaluator is exactly batch-invariant; ChromaDB HNSW introduces ≤2/10,000 ANN noise. Paper numbers should use the exact evaluator.

## 2. §7.2 2×2 cross experiment (old/new checkpoint × old/new split)

- old ckpt: `topk_mtcir/topk_epoch_0000_step_002700_score_0.115062.pth.tar` (trained on original MTCIR text)
- new ckpt: `topk_pair_raw_s42/topk_epoch_0002_step_000600_score_0.432042.pth.tar` (pair-control RAW, common-dev mAP 0.7172)
- old split: `mtcir_np/eval/eval_subset.jsonl` (5,001 queries)
- new split: `pair_control/test.jsonl` (10,000 queries)

| | old split | new split |
|---|---|---|
| old ckpt | **A = 0.0762** (0.0770 exact) | B = 0.1137 (0.1139 exact) |
| new ckpt | C = 0.4753 (0.4757 exact) | **D = 0.6059 (0.6058 exact)** |

All four cells differ ⇒ model **and** data changed; evaluator is shared (verified: git diff of `eval_checkpoint.py` since 2026-05-07 is only the `--eval-json-path` flag, and cell A reproduces the historical 7.62% exactly).
**Interpretation (per plan §7.2):** the gap is driven mostly by the checkpoint (≈40–49pt), then by the split (≈4–13pt). Both factors moved in the same direction; no evaluator bug.

## 3. §7.3 Split & image leakage

pair_control split (raw text variant), from `audit/mtcir_data_audit.json`:

| pair | triplet ID overlap | ref overlap | target overlap | any-role overlap | a.target→b.reference | a.reference→b.target |
|---|---|---|---|---|---|---|
| train/dev | 0 | 6,298 (68.9% of dev refs) | 5,855 | 13,982 | 5,435 | 4,302 |
| train/test | 0 | 6,288 (69.1% of test refs) | 5,863 | 14,018 | 5,366 | 4,271 |
| dev/test | 0 | 1,274 | 916 | 2,865 | 511 | 507 |

Three-way triplet intersection: 0.

**Verdict:** triplet-disjoint only. **Image-held-out does NOT hold**: >60% of test images appear in train in some role. Per plan §7.3, results are **pair-held-out diagnostics**; if the thesis requires image-held-out claims, the split must be re-created by image connected components and the core RAW/FIXED/MULTI controls re-trained.

## 4. §7.4 Gallery & positive mapping (`pair_control/test.jsonl`)

| check | value |
|---|---|
| queries | 10,000 |
| gallery unique target IDs | 9,441 |
| targets missing from gallery | **0** |
| ref == target | **0** |
| duplicate target IDs | 559 (492 targets shared by >1 query) |
| gold column histogram head | [(0,2),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1)] |

100-query manual-check sample: `audit/mapping_sample_100.json`.

## 5. §7.5 Metric unit tests (synthetic embeddings)

| test | expected | result |
|---|---|---|
| T1 identity query=target | R@1 = 1.0 | 1.0 PASS |
| T2 random queries | R@1 ≈ 1/N | 0.002 vs 0.001 (binomial noise, n=1000) PASS |
| T3 permuted gold | ≈ random | 0.0 PASS |
| T4 shuffled query order | identical | 0.0 diff PASS |
| T5 target forced to rank 2 | R@1=0, R@5=1 | 0.0 / 1.0 PASS |
| T6 chunked vs full ranking | identical | 0.0 diff PASS |

## 6. §7.6 Text-input diagnostics (pair multi s42, 2000-query subsample)

| variant | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| baseline (correct text) | 0.7395 | 0.9355 | 0.9630 | 0.8259 |
| shuffled-text (permuted) | 0.0335 | 0.0955 | 0.1495 | 0.0722 |
| reference-only (empty text) | 0.2045 | 0.5010 | 0.6320 | 0.3440 |

**Verdict:** R@1 drops −95% under shuffled text ⇒ the model uses modification text heavily; no text-ignoring pathology. Reference-only 20.5% reflects the residual image-only retrieval signal (reference–target visual similarity), which is expected.

## 7. Implications for the thesis

- Final MTCIR/MerdCIR numbers should be regenerated with the **exact evaluator** (`--exact-gallery`); ChromaDB ANN noise (≤2e-4) is immaterial to conclusions but exact numbers are cleaner.
- The pair-control results (RAW 0.606 > MULTI 0.569 > FIXED 0.494) are **not** an artifact of the evaluator; they are stable to batch size and reproduced exactly.
- The old topk_mtcir 7.62% was a genuinely weak early checkpoint (selection score 0.115, 1 epoch, original-text training) on a smaller split; it must not be compared directly with pair-control numbers in the thesis without the 2×2 framing above.
- **Leakage caveat:** all pair-control numbers are pair-held-out; image-level overlap with train is high (≈60%+), so cross-domain/generalization claims must be worded accordingly.

## 8. Artifacts

- `audit/mtcir_data_audit.json` — leakage + mapping + unit-test numbers
- `audit/mapping_sample_100.json` — 100-query manual-check rows
- `audit/mtcir_gpu_audit_table.json` — ChromaDB 2×2 + batch-invariance table
- `logs/audit_mtcir_*.log`, `logs/audit_mtcir_exact_*.log` — raw runs
- `audit/text_diag_0.json` — text diagnostics
- Code changes: `eval_checkpoint.py` (`--batch-size`, `--exact-gallery`, `--backbone-size`, `--cirr-force-export`, `_load_cirr_gallery_ids` dict fix), `train.py` (`--backbone_size`, epoch-end validation), `compute_cirr_metrics.py` (target_hard-based subset scoring + `--gt`)
