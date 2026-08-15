#!/bin/bash
# 2x2 capacity pilot evaluation (plan §8.6): 4 conditions x 5 datasets.
# Conditions: Fixed-L / Multi-L / Fixed-B96 / Multi-B96 (all seed 42, batch 96,
# step-2400 matched checkpoints). Uses the exact evaluator for MTCIR/MerdCIR.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/vitl_2x2_eval_${TS}.log"
mkdir -p ./logs

pick_step_ckpt() {
    # pick_step_ckpt <topk_json> <step>  -> prints the checkpoint path for that step
    $PYTHON - "$1" "$2" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    recs = json.load(f)
step = int(sys.argv[2])
for r in sorted(recs, key=lambda x: x["step"]):
    if r["step"] == step:
        print(r["path"])
        sys.exit(0)
# fallback: nearest step below
nearest = min(recs, key=lambda x: abs(x["step"] - step))
print(nearest["path"], file=sys.stderr)
print(nearest["path"])
PYEOF
}

eval_one() {
    local TAG=$1 CKPT=$2 BB=$3
    echo "" | tee -a "$LOG"
    echo "========== ${TAG} (${CKPT}, backbone=${BB}) ==========" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset MTCIR --method cross_attn_alpha \
        --backbone-size "$BB" --exact-gallery --eval-json-path pair_control/test.jsonl \
        2>&1 | grep -E "Dataset:|Recall@|mAP" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset MerdCIR --method cross_attn_alpha \
        --backbone-size "$BB" --exact-gallery \
        2>&1 | grep -E "Dataset:|Recall@|mAP" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset FashionIQ --method cross_attn_alpha \
        --backbone-size "$BB" \
        2>&1 | grep -E "Dataset:|Recall@|mAP" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method cross_attn_alpha \
        --backbone-size "$BB" --cirr-metric recall \
        --cirr-json-path cap.rc2.val.json --cirr-split-json-path split.rc2.val.json \
        2>&1 | grep -E "Dataset:|Recall@|mAP" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method cross_attn_alpha \
        --backbone-size "$BB" --cirr-metric recall_subset \
        --cirr-json-path cap.rc2.val.json --cirr-split-json-path split.rc2.val.json \
        2>&1 | grep -E "Dataset:|Recall@|mAP" | tee -a "$LOG"
}

STEP=2400

FIXED_L_CKPT="./checkpoints/topk_pair_fixed_L_s42/step2400.pth.tar"
eval_one "Fixed-L" "$FIXED_L_CKPT" "L"

MULTI_L_CKPT=$(pick_step_ckpt ./checkpoints/topk_checkpoints_multi_L_s42.json $STEP)
eval_one "Multi-L" "$MULTI_L_CKPT" "L"

FIXED_B96_CKPT=$(pick_step_ckpt ./checkpoints/topk_checkpoints_fixed_B96_s42.json $STEP)
eval_one "Fixed-B96" "$FIXED_B96_CKPT" "B"

MULTI_B96_CKPT=$(pick_step_ckpt ./checkpoints/topk_checkpoints_multi_B96_s42.json $STEP)
eval_one "Multi-B96" "$MULTI_B96_CKPT" "B"

echo "" | tee -a "$LOG"
echo "2x2 evaluation complete: $LOG" | tee -a "$LOG"
