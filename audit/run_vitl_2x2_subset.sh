#!/bin/bash
# Re-run ONLY the CIRR recall_subset evals for the 2x2 matrix.
# First pass's grep pattern missed the "Recall_subset@K" output format.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/vitl_2x2_subset_${TS}.log"
mkdir -p ./logs

pick_step_ckpt() {
    $PYTHON - "$1" "$2" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    recs = json.load(f)
step = int(sys.argv[2])
for r in sorted(recs, key=lambda x: x["step"]):
    if r["step"] == step:
        print(r["path"])
        sys.exit(0)
nearest = min(recs, key=lambda x: abs(x["step"] - step))
print(nearest["path"])
PYEOF
}

subset_eval() {
    local TAG=$1 CKPT=$2 BB=$3
    echo "========== ${TAG} recall_subset (${CKPT}, backbone=${BB}) ==========" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method cross_attn_alpha \
        --backbone-size "$BB" --cirr-metric recall_subset \
        --cirr-json-path cap.rc2.val.json --cirr-split-json-path split.rc2.val.json \
        2>&1 | grep -E "Dataset:|Recall_subset@|Recall@|mAP" | tee -a "$LOG"
}

STEP=2400
subset_eval "Fixed-L" "./checkpoints/topk_pair_fixed_L_s42/step2400.pth.tar" "L"
subset_eval "Multi-L" "$(pick_step_ckpt ./checkpoints/topk_checkpoints_multi_L_s42.json $STEP)" "L"
subset_eval "Fixed-B96" "$(pick_step_ckpt ./checkpoints/topk_checkpoints_fixed_B96_s42.json $STEP)" "B"
subset_eval "Multi-B96" "$(pick_step_ckpt ./checkpoints/topk_checkpoints_multi_B96_s42.json $STEP)" "B"

echo "subset-only evaluation complete: $LOG" | tee -a "$LOG"
