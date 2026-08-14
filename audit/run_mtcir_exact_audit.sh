#!/bin/bash
# Exact-gallery confirmation (plan §7.1): batch-size invariance must be EXACT with
# matmul ranking; re-confirm the 2x2 cells with the exact evaluator.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/audit_mtcir_exact_${TS}.log"
mkdir -p ./logs

OLD_CKPT="./checkpoints/topk_mtcir/topk_epoch_0000_step_002700_score_0.115062.pth.tar"
NEW_CKPT="./checkpoints/topk_pair_raw_s42/topk_epoch_0002_step_000600_score_0.432042.pth.tar"
OLD_SPLIT="mtcir_np/eval/eval_subset.jsonl"
NEW_SPLIT="pair_control/test.jsonl"
METHOD="cross_attn_alpha"

run_eval() {
    local TAG=$1 CKPT=$2 SPLIT=$3 BS=$4
    echo "=== $TAG ===" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset MTCIR --method "$METHOD" \
        --eval-json-path "$SPLIT" --batch-size "$BS" --exact-gallery 2>&1 | tee -a "$LOG"
}

echo "[$(date '+%F %T')] exact-gallery audit start" | tee -a "$LOG"

# batch invariance (exact): must be IDENTICAL across batch sizes
for BS in 64 128 256; do
    run_eval "exact_batchinv_bs${BS}" "$NEW_CKPT" "$NEW_SPLIT" "$BS"
done

# 2x2 with exact evaluator
run_eval "exact_x2_A_oldckpt_oldsplit" "$OLD_CKPT" "$OLD_SPLIT" 128
run_eval "exact_x2_B_oldckpt_newsplit" "$OLD_CKPT" "$NEW_SPLIT" 128
run_eval "exact_x2_C_newckpt_oldsplit" "$NEW_CKPT" "$OLD_SPLIT" 128
run_eval "exact_x2_D_newckpt_newsplit" "$NEW_CKPT" "$NEW_SPLIT" 128

echo "[$(date '+%F %T')] exact-gallery audit complete" | tee -a "$LOG"
