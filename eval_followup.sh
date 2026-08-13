#!/bin/bash
# Evaluate all pair_control checkpoints on CIRR, FashionIQ, MTCIR test, and MerdCIR test
# Run AFTER all training completes.
#
# Checkpoint selection follows the pre-registered protocol (FOLLOWUP_EXPERIMENT_PLAN.md §5.1):
#   c* = argmax_c mAP_common-dev(c), picked from the training_log per condition.
#   The training-time selection_score (0.2*dev + 0.8*CIRR) is NOT used for final selection.
#   If the common-dev-best checkpoint file was not retained by topk (unlikely), fall back
#   to the best retained file by common-dev mAP.

set -e

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_LOG="./logs/eval_followup_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$EVAL_LOG"
}

eval_checkpoint() {
    local CKPT=$1
    local DATASET=$2
    local METHOD=$3
    local OUTPUT=$4
    local EXTRA_ARGS=$5

    log "  Evaluating: $CKPT on $DATASET"
    $PYTHON eval_checkpoint.py \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --output-json "$OUTPUT" \
        $EXTRA_ARGS \
        2>&1 | tee -a "$EVAL_LOG"
}

# Select the best checkpoint for a condition: argmax common-dev mAP over its training_log,
# preferring an entry whose checkpoint file still exists; else best retained file.
select_best_checkpoint() {
    local CKPT_DIR=$1
    local TRAINING_LOG=$2

    $PYTHON - "$CKPT_DIR" "$TRAINING_LOG" <<'PYEOF'
import json, os, sys
ckpt_dir, training_log = sys.argv[1], sys.argv[2]

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

log_entries = load(training_log)
if log_entries:
    ranked = sorted(log_entries, key=lambda e: e.get("in_domain", {}).get("mAP", 0.0), reverse=True)
    for e in ranked:
        path = os.path.join(ckpt_dir, f"topk_epoch_{e['epoch']:04d}_step_{e['step']:06d}_score_{e['selection_score']:.6f}.pth.tar")
        if os.path.isfile(path):
            print(path)
            sys.exit(0)
    # fallback: none of the common-dev-best files retained; find best retained by filename score
    import glob
    files = glob.glob(os.path.join(ckpt_dir, "topk_epoch_*.pth.tar"))
    if files:
        print(sorted(files)[-1])
        sys.exit(0)
print("")
PYEOF
}

# All six conditions (dir + training_log pairs)
CONDITIONS=(
    "checkpoints/topk_pair_raw_s42 checkpoints/training_log_raw_s42.json"
    "checkpoints/topk_pair_raw_s123 checkpoints/training_log_raw_s123.json"
    "checkpoints/topk_pair_fixed_s42 checkpoints/training_log_fixed_s42.json"
    "checkpoints/topk_pair_fixed_s123 checkpoints/training_log_fixed_s123.json"
    "checkpoints/topk_pair_multi_s42 checkpoints/training_log_multi_s42.json"
    "checkpoints/topk_pair_multi_s123 checkpoints/training_log_multi_s123.json"
)

METHOD="cross_attn_alpha"

log "============================================================"
log "EVALUATION PIPELINE START"
log "============================================================"

for COND in "${CONDITIONS[@]}"; do
    set -- $COND
    CKPT_DIR=$1
    TRAINING_LOG=$2

    if [ ! -d "$CKPT_DIR" ]; then
        log "SKIP: $CKPT_DIR not found"
        continue
    fi

    BEST_CKPT=$(select_best_checkpoint "$CKPT_DIR" "$TRAINING_LOG")
    if [ -z "$BEST_CKPT" ]; then
        log "WARNING: No checkpoint selected for $CKPT_DIR"
        continue
    fi

    CKPT_NAME=$(basename "$CKPT_DIR")
    log ""
    log "--- Evaluating $CKPT_NAME ---"
    log "  Selected checkpoint (common-dev mAP): $BEST_CKPT"

    # MTCIR source-domain held-out test (frozen pair_control/test.jsonl, plan §7.1)
    eval_checkpoint "$BEST_CKPT" "MTCIR" "$METHOD" \
        "${CKPT_DIR}/${CKPT_NAME}_mtcir_test.json" \
        "--eval-json-path pair_control/test.jsonl"

    # MerdCIR eval (comparison)
    eval_checkpoint "$BEST_CKPT" "MerdCIR" "$METHOD" \
        "${CKPT_DIR}/${CKPT_NAME}_merdcir_eval.json" \
        ""

    # FashionIQ
    eval_checkpoint "$BEST_CKPT" "FashionIQ" "$METHOD" \
        "${CKPT_DIR}/${CKPT_NAME}_fashioniq.json" \
        ""

    # CIRR recall
    eval_checkpoint "$BEST_CKPT" "CIRR" "$METHOD" \
        "${CKPT_DIR}/${CKPT_NAME}_cirr_recall.json" \
        "--cirr-metric recall"

    # CIRR recall_subset
    eval_checkpoint "$BEST_CKPT" "CIRR" "$METHOD" \
        "${CKPT_DIR}/${CKPT_NAME}_cirr_recall_subset.json" \
        "--cirr-metric recall_subset"

done

log ""
log "============================================================"
log "EVALUATION PIPELINE COMPLETE"
log "============================================================"
