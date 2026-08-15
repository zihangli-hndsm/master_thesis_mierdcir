#!/bin/bash
# Re-run ONLY the CIRR val evaluations for the 2x2 matrix (the first pass used the
# default test1 json path and produced no output). Explicitly points at the val JSONs.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/vitl_2x2_cirr_${TS}.log"
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

cirr_eval() {
    local TAG=$1 CKPT=$2 BB=$3 METRIC=$4
    echo "========== ${TAG} ${METRIC} (${CKPT}, backbone=${BB}) ==========" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method cross_attn_alpha \
        --backbone-size "$BB" --cirr-metric "$METRIC" \
        --cirr-json-path cap.rc2.val.json --cirr-split-json-path split.rc2.val.json \
        2>&1 | grep -E "Dataset:|Recall@|mAP" | tee -a "$LOG"
}

STEP=2400
FIXED_L="./checkpoints/topk_pair_fixed_L_s42/step2400.pth.tar"
MULTI_L=$(pick_step_ckpt ./checkpoints/topk_checkpoints_multi_L_s42.json $STEP)
FIXED_B96=$(pick_step_ckpt ./checkpoints/topk_checkpoints_fixed_B96_s42.json $STEP)
MULTI_B96=$(pick_step_ckpt ./checkpoints/topk_checkpoints_multi_B96_s42.json $STEP)

cirr_eval "Fixed-L" "$FIXED_L" "L" "recall"
cirr_eval "Fixed-L" "$FIXED_L" "L" "recall_subset"
cirr_eval "Multi-L" "$MULTI_L" "L" "recall"
cirr_eval "Multi-L" "$MULTI_L" "L" "recall_subset"
cirr_eval "Fixed-B96" "$FIXED_B96" "B" "recall"
cirr_eval "Fixed-B96" "$FIXED_B96" "B" "recall_subset"
cirr_eval "Multi-B96" "$MULTI_B96" "B" "recall"
cirr_eval "Multi-B96" "$MULTI_B96" "B" "recall_subset"

echo "CIRR-only evaluation complete: $LOG" | tee -a "$LOG"
