# CIRR test1 Submission Guide

**Status:** pipeline ready (exporter + validator + rehearsal passed); submission pending ViT-L pilot decision (§10).
**Date:** 2026-08-13

## 1. Server

- Main: http://cirr.cecs.anu.edu.au/ (back online May 2025)
- Backup: https://cirr.junjie.au/ (use if main is down)
- Requires an account / email (hCaptcha since May 2025).

## 2. Files to upload (per model)

Two independent JSON files per checkpoint, uploaded separately:

| metric | file | content per query |
|---|---|---|
| Recall | `<cond>_<seed>_recall.json` | top-50 image IDs from the full test1 gallery |
| RecallSubset | `<cond>_<seed>_recall_subset.json` | top-3 from `img_set.members` |

JSON schema (verified against official template in `Test-split_server.md`):

```json
{
  "version": "rc2",
  "metric": "recall",            // or "recall_subset"
  "12063": ["test1-233-3-img1", "test1-969-1-img0", ...]
}
```

## 3. Generation command

```bash
PY=/home/zihali/data/conda/envs/thesis/bin/python3
$PY eval_checkpoint.py \
  --checkpoint <ckpt.pth.tar> \
  --dataset CIRR --method cross_attn_alpha --backbone-size B \
  --cirr-metric recall \
  --output-json cirr_test_results/<cond>_<seed>_recall.json
# repeat with --cirr-metric recall_subset
```

The test1 gallery comes from `data/CIRR/split.rc2.test1.json` (2315 images; all present in LMDB). Batch size does not affect output; fixed gallery order + shuffle=False makes generation deterministic.

## 4. Validation before upload

```bash
$PY audit/cirr_validate.py --pred <file>.json \
  --gt data/CIRR/cap.rc2.test1.json \
  --split data/CIRR/split.rc2.test1.json \
  --checkpoint <ckpt.pth.tar>
# expect OVERALL: PASS; writes <file>.json.validation.json manifest (SHA256 + command)
```

Checks: version/metric, key coverage (4148 pairids), exactly 50/3 unique candidates, candidates ∈ split, no reference, subset ⊆ members, < 5 MB.

## 5. Upload order & recording

1. Upload `recall` file first, then `recall_subset` for the same checkpoint.
2. Save the server-returned Recall@K / RecallSubset@K per file into `server_results.csv` (template below).
3. Do NOT re-select checkpoints based on test results (§2.2).

## 6. Recording templates

`server_results.csv`:

```csv
submission_date,condition,seed,metric,file_sha256,server_R1,server_R5,server_R10,server_R50,notes
2026-08-13,multi,s42,recall,<sha256>,,,,,
2026-08-13,multi,s42,recall_subset,<sha256>,,,,,
```

`submission_manifest.json` (aggregate of the per-file `.validation.json` manifests):

```json
{
  "protocol": "NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md §2-§6",
  "checkpoint_selection": "common-dev mAP (pre-registered); test never used",
  "submissions": [
    {"condition": "multi", "seed": 42, "metric": "recall",
     "file": "cirr_test_results/multi_s42_recall.json",
     "sha256": "...", "validator": "PASS", "server_results": {}},
    ...
  ]
}
```

## 7. Frozen checkpoint table

See `cirr_test_results/checkpoint_manifest.json` (9 ViT-B checkpoints; SHA256 recorded). Final submission set may change after the ViT-L pilot (§8.7 trigger).
