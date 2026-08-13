#!/bin/bash
# Follow-up training - PRIORITY ORDER (for time-constrained 13.5h session)
# Order: RAW s42 (already running) → MULTI s42 → FIXED s42 → RAW s123 → MULTI s123 → FIXED s123
# This ensures all conditions get at least 1 seed before any gets a second seed.
# FIXED s123 is lowest priority and will be skipped if time runs out.

set -e

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

COMMON_ARGS="
  --method merdcir_mlp_alpha
  --lmdb_path ./data/MTCIR/images_224_lmdb
  --cirr_data_path ./data/CIRR
  --cirr_json_path cap.rc2.val.json
  --cirr_split_json_path split.rc2.val.json
  --cirr_lmdb_path ./data/CIRR/images_224_lmdb
  --eval_json_path pair_control/dev.jsonl
  --resume_path ./checkpoints/_no_resume.pth.tar
  --epochs 3
  --batch_size 300
"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="./logs/followup_v2_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

run_training() {
    local CONDITION=$1 SEED=$2 TRAIN_JSON=$3 CKPT_DIR=$4 TRAINING_LOG=$5 TOPK_JSON=$6 PRIORITY=$7

    # Check remaining time
    REMAINING=$(python3 -c "
import json, time
try:
    with open('${TRAINING_LOG}') as f:
        d = json.load(f)
    print('done')
except:
    print('pending')
" 2>/dev/null)

    if [ "$REMAINING" = "done" ]; then
        log "SKIP: ${CONDITION} seed=${SEED} already completed"
        return 0
    fi

    log "============================================================"
    log "[PRIORITY ${PRIORITY}] STARTING: ${CONDITION} seed=${SEED}"
    log "  Train data: ${TRAIN_JSON}"
    log "============================================================"

    $PYTHON train.py \
        $COMMON_ARGS \
        --merdcir_json_path "${TRAIN_JSON}" \
        --topk_checkpoint_dir "${CKPT_DIR}" \
        --training_log_path "${TRAINING_LOG}" \
        --topk_json_path "${TOPK_JSON}" \
        --seed "${SEED}" \
        2>&1 | tee -a "$MAIN_LOG"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: ${CONDITION} seed=${SEED} failed with exit code ${EXIT_CODE}"
        return $EXIT_CODE
    fi
    log "COMPLETED: ${CONDITION} seed=${SEED}"
}

mkdir -p ./checkpoints ./logs

log "============================================================"
log "FOLLOW-UP V2 (PRIORITY ORDER) START"
log "============================================================"

# Priority 1: RAW s42 (already running/completed)
log ""
log "--- ROUND 1: one seed per condition ---"

run_training "RAW" 42 \
    "pair_control/train.jsonl" \
    "./checkpoints/topk_pair_raw_s42" \
    "./checkpoints/training_log_raw_s42.json" \
    "./checkpoints/topk_checkpoints_raw_s42.json" \
    "1"

run_training "MULTI" 42 \
    "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_s42" \
    "./checkpoints/training_log_multi_s42.json" \
    "./checkpoints/topk_checkpoints_multi_s42.json" \
    "2"

run_training "FIXED" 42 \
    "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_s42" \
    "./checkpoints/training_log_fixed_s42.json" \
    "./checkpoints/topk_checkpoints_fixed_s42.json" \
    "3"

# Priority 4-6: second seeds (if time permits)
log ""
log "--- ROUND 2: second seeds (time permitting) ---"

run_training "RAW" 123 \
    "pair_control/train.jsonl" \
    "./checkpoints/topk_pair_raw_s123" \
    "./checkpoints/training_log_raw_s123.json" \
    "./checkpoints/topk_checkpoints_raw_s123.json" \
    "4"

run_training "MULTI" 123 \
    "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_s123" \
    "./checkpoints/training_log_multi_s123.json" \
    "./checkpoints/topk_checkpoints_multi_s123.json" \
    "5"

run_training "FIXED" 123 \
    "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_s123" \
    "./checkpoints/training_log_fixed_s123.json" \
    "./checkpoints/topk_checkpoints_fixed_s123.json" \
    "6"

log ""
log "============================================================"
log "ALL PRIORITIZED TRAINING COMPLETE"
log "============================================================"
