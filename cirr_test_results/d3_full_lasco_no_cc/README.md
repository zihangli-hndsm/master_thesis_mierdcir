# D3-full LaSCo CIRR test exports

This directory contains six locally generated `rc2` test1 prediction files:
three frozen D3-full LaSCo/no-Text-CC checkpoints, each with global recall and
candidate-subset recall exports.

All six files passed `audit/cirr_validate.py`, including key coverage,
candidate count, split membership, reference exclusion, and subset membership.
The files have **not** been uploaded to the CIRR server. Therefore
`server_results_template.csv` intentionally contains no server metrics;
`submission_manifest.json` records `server_status: not_submitted`.

Generation entry point:

```bash
cd /home/zihali/final_thesis/reproduction_experiments/ConText-CIR-09052026
python src/make_cirr_sub.py \
  --path reproduction_evidence/session_d3_full_lasco_no_cc_s0/epoch=0-step=1623-val_loss=2.192e+00.ckpt \
  --backbone_size B \
  --submission_name d3_full_lasco_no_cc_s0 \
  --output_dir /home/zihali/final_thesis/cirr_test_results/d3_full_lasco_no_cc
```

The seed-1 and seed-2 commands use their corresponding frozen checkpoint
paths. Checkpoint and prediction hashes are in `submission_manifest.json`.
