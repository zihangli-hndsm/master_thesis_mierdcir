#!/bin/bash
# Evaluate the 3rd seed (s2025) pair_control checkpoints on all 5 eval datasets.
# MTCIR test = frozen pair_control/test.jsonl; CIRR = val set direct evaluation (has labels).
# Log format is parsed by merge_s2025_metrics.py: "=== <cond> ===" + "Dataset: X" + metric lines.
# Usage: bash eval_s2025.sh            # evaluate all s2025 conditions with existing checkpoints
#        bash eval_s2025.sh multi_s2025 fixed_s2025 raw_s2025

set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_LOG="./logs/eval_s2025_${TIMESTAMP}.log"
METHOD="cross_attn_alpha"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$EVAL_LOG"
}

# Select best checkpoint for a condition: argmax in_domain mAP over its training_log
select_best_checkpoint() {
    local CKPT_DIR=$1
    local TRAINING_LOG=$2
    $PYTHON - "$CKPT_DIR" "$TRAINING_LOG" <<'PYEOF'
import json, os, sys
ckpt_dir, training_log = sys.argv[1], sys.argv[2]
if not os.path.isfile(training_log):
    sys.exit("")
entries = json.load(open(training_log))
ranked = sorted(entries, key=lambda e: e.get("in_domain", {}).get("mAP", 0.0), reverse=True)
for e in ranked:
    path = os.path.join(ckpt_dir, f"topk_epoch_{e['epoch']:04d}_step_{e['step']:06d}_score_{e['selection_score']:.6f}.pth.tar")
    if os.path.isfile(path):
        print(path)
        sys.exit(0)
print("")
PYEOF
}

eval_condition() {
    local COND=$1 CKPT=$2
    log "=== $COND ==="
    log "Selected checkpoint: $CKPT"

    log "Evaluating: $CKPT on MTCIR"
    $PYTHON eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset MTCIR --method "$METHOD" \
        --eval-json-path pair_control/test.jsonl 2>&1 | tee -a "$EVAL_LOG"

    log "Evaluating: $CKPT on MerdCIR"
    $PYTHON eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset MerdCIR --method "$METHOD" 2>&1 | tee -a "$EVAL_LOG"

    log "Evaluating: $CKPT on FashionIQ"
    $PYTHON eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset FashionIQ --method "$METHOD" 2>&1 | tee -a "$EVAL_LOG"

    log "Evaluating: $CKPT on CIRR (val recall)"
    $PYTHON eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
        --cirr-metric recall --cirr-json-path cap.rc2.val.json 2>&1 | tee -a "$EVAL_LOG"

    log "Evaluating: $CKPT on CIRR (val recall_subset)"
    $PYTHON eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
        --cirr-metric recall_subset --cirr-json-path cap.rc2.val.json 2>&1 | tee -a "$EVAL_LOG"
}

CONDITIONS=("$@")
if [ ${#CONDITIONS[@]} -eq 0 ]; then
    CONDITIONS=(multi_s2025 fixed_s2025 raw_s2025)
fi

mkdir -p ./logs
log "============================================================"
log "S2025 EVALUATION START (conditions: ${CONDITIONS[*]})"
log "============================================================"

for COND in "${CONDITIONS[@]}"; do
    CKPT_DIR="./checkpoints/topk_pair_${COND}"
    TRAINING_LOG="./checkpoints/training_log_${COND}.json"
    if [ ! -d "$CKPT_DIR" ]; then
        log "SKIP: $CKPT_DIR not found"
        continue
    fi
    BEST_CKPT=$(select_best_checkpoint "$CKPT_DIR" "$TRAINING_LOG")
    if [ -z "$BEST_CKPT" ]; then
        log "WARNING: No checkpoint selected for $CKPT_DIR"
        continue
    fi
    eval_condition "$COND" "$BEST_CKPT"
done

log "============================================================"
log "S2025 EVALUATION COMPLETE"
log "============================================================"
