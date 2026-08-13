#!/bin/bash
# Re-evaluate all frozen pair-control checkpoints with the EXACT evaluator
# (--exact-gallery) for final thesis numbers (audit recommendation).
# MTCIR (pair_control/test.jsonl) + MerdCIR (merdcir_np/eval/eval_subset.jsonl)
# Run AFTER training is done (needs GPU; ~2-4 min per (ckpt, dataset)).
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/exact_reeval_${TS}.log"
mkdir -p ./logs

CONDS="raw_s42 raw_s123 raw_s2025 fixed_s42 fixed_s123 fixed_s2025 multi_s42 multi_s123 multi_s2025"
METHOD="cross_attn_alpha"

echo "[$(date '+%F %T')] exact re-eval start" | tee -a "$LOG"
for COND in $CONDS; do
    CKPT=$(/home/zihali/data/conda/envs/thesis/bin/python3 - <<PYEOF
import json
m = json.load(open("cirr_test_results/checkpoint_manifest.json"))
print(m["checkpoints"]["$COND"]["checkpoint"])
PYEOF
)
    echo "=== ${COND} MTCIR (exact) ===" | tee -a "$LOG"
    $PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset MTCIR --method "$METHOD" \
        --eval-json-path pair_control/test.jsonl --exact-gallery 2>&1 | grep -E "Dataset:|Recall|mAP" | tee -a "$LOG"
    echo "=== ${COND} MerdCIR (exact) ===" | tee -a "$LOG"
    $PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset MerdCIR --method "$METHOD" \
        --exact-gallery 2>&1 | grep -E "Dataset:|Recall|mAP" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] exact re-eval complete" | tee -a "$LOG"
