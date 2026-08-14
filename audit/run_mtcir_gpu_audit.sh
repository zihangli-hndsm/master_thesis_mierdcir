#!/bin/bash
# MTCIR evaluation audit (plan §7): batch-size invariance + 2x2 old/new cross experiment.
# Log format is parsed by audit/parse_audit_logs.py.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/audit_mtcir_${TS}.log"
mkdir -p ./logs

OLD_CKPT="./checkpoints/topk_mtcir/topk_epoch_0000_step_002700_score_0.115062.pth.tar"
NEW_CKPT="./checkpoints/topk_pair_raw_s42/topk_epoch_0002_step_000600_score_0.432042.pth.tar"
OLD_SPLIT="mtcir_np/eval/eval_subset.jsonl"     # old 5K eval subset (source of R@1=7.62)
NEW_SPLIT="pair_control/test.jsonl"             # frozen 10K pair-control split (source of R@1=60.62)
METHOD="cross_attn_alpha"

run_eval() {
    local TAG=$1 CKPT=$2 SPLIT=$3 BS=$4
    echo "=== $TAG ===" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset MTCIR --method "$METHOD" \
        --eval-json-path "$SPLIT" --batch-size "$BS" 2>&1 | tee -a "$LOG"
}

echo "[$(date '+%F %T')] MTCIR GPU audit start (old ckpt: $(basename $OLD_CKPT), new ckpt: $(basename $NEW_CKPT))" | tee -a "$LOG"

# ---- §7.1 batch-size invariance (new ckpt, new split) ----
for BS in 32 64 128 256; do
    run_eval "batch_invariance_bs${BS}_newckpt_newsplit" "$NEW_CKPT" "$NEW_SPLIT" "$BS"
done

# ---- §7.2 2x2 cross experiment ----
# A: old ckpt x old split  (expected ~7.62, reproduces historical number)
run_eval "x2_A_oldckpt_oldsplit" "$OLD_CKPT" "$OLD_SPLIT" 128
# B: old ckpt x new split
run_eval "x2_B_oldckpt_newsplit" "$OLD_CKPT" "$NEW_SPLIT" 128
# C: new ckpt x old split
run_eval "x2_C_newckpt_oldsplit" "$NEW_CKPT" "$OLD_SPLIT" 128
# D: new ckpt x new split  (expected ~60.62, reproduces pair-control number)
run_eval "x2_D_newckpt_newsplit" "$NEW_CKPT" "$NEW_SPLIT" 128

echo "[$(date '+%F %T')] MTCIR GPU audit complete" | tee -a "$LOG"
