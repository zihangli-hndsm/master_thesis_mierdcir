#!/bin/bash
# Follow-up experiment training pipeline
# Runs Raw, Fixed, Multi training sequentially on pair_control data
# Each run: 3 epochs, ~2.1 hours
# Total budget: ~13 hours for 6 runs (2 seeds × 3 conditions)

set -e

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

# Common args
COMMON_ARGS="
  --method merdcir_mlp_alpha
  --lmdb_path ./data/MTCIR/images_224_lmdb
  --cirr_data_path ./data/CIRR
  --cirr_json_path cap.rc2.val.json
  --cirr_split_json_path split.rc2.val.json
  --cirr_lmdb_path ./data/CIRR/images_224_lmdb
  --eval_json_path pair_control/dev.jsonl
  --epochs 3
  --batch_size 300
"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="./logs/followup_experiments_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

run_training() {
    local CONDITION=$1
    local SEED=$2
    local TRAIN_JSON=$3
    local CKPT_DIR=$4
    local TRAINING_LOG=$5
    local TOPK_JSON=$6

    log "============================================================"
    log "STARTING: ${CONDITION} seed=${SEED}"
    log "  Train data: ${TRAIN_JSON}"
    log "  Checkpoint dir: ${CKPT_DIR}"
    log "  Seed: ${SEED}"
    log "============================================================"

    $PYTHON scripts/train.py \
        $COMMON_ARGS \
        --merdcir_json_path "${TRAIN_JSON}" \
        --topk_checkpoint_dir "${CKPT_DIR}" \
        --training_log_path "${TRAINING_LOG}" \
        --topk_json_path "${TOPK_JSON}" \
        --resume_path ./checkpoints/_no_resume.pth.tar \
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
log "FOLLOW-UP EXPERIMENT PIPELINE START"
log "Timestamp: ${TIMESTAMP}"
log "============================================================"

# --- Seed 42 series ---
log ""
log "=== SEED 42 SERIES ==="

run_training "RAW" 42 \
    "pair_control/train.jsonl" \
    "./checkpoints/topk_pair_raw_s42" \
    "./checkpoints/training_log_raw_s42.json" \
    "./checkpoints/topk_checkpoints_raw_s42.json"

run_training "FIXED" 42 \
    "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_s42" \
    "./checkpoints/training_log_fixed_s42.json" \
    "./checkpoints/topk_checkpoints_fixed_s42.json"

run_training "MULTI" 42 \
    "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_s42" \
    "./checkpoints/training_log_multi_s42.json" \
    "./checkpoints/topk_checkpoints_multi_s42.json"

# --- Seed 123 series ---
log ""
log "=== SEED 123 SERIES ==="

run_training "RAW" 123 \
    "pair_control/train.jsonl" \
    "./checkpoints/topk_pair_raw_s123" \
    "./checkpoints/training_log_raw_s123.json" \
    "./checkpoints/topk_checkpoints_raw_s123.json"

run_training "FIXED" 123 \
    "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_s123" \
    "./checkpoints/training_log_fixed_s123.json" \
    "./checkpoints/topk_checkpoints_fixed_s123.json"

run_training "MULTI" 123 \
    "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_s123" \
    "./checkpoints/training_log_multi_s123.json" \
    "./checkpoints/topk_checkpoints_multi_s123.json"

log ""
log "============================================================"
log "ALL TRAINING RUNS COMPLETED"
log "============================================================"
