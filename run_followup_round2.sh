#!/bin/bash
# Round-2 pair_control training: complete remaining conditions.
# Priority order (v2): RAW s42 (done) -> MULTI s42 -> FIXED s42 -> RAW s123 -> MULTI s123 -> FIXED s123
# FIXED s123 is lowest priority: run last, may be skipped if time runs out.
# A condition is skipped if its topk_json already exists (non-empty).
# A failed run does NOT abort the pipeline (no set -e); the script continues to the next condition.

set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

# Identical to the RAW s42 run (run_followup_experiments.sh) except data/seed/dirs.
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
MAIN_LOG="./logs/followup_round2_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

run_training() {
    local CONDITION=$1 SEED=$2 TRAIN_JSON=$3 CKPT_DIR=$4 TRAINING_LOG=$5 TOPK_JSON=$6
    if [ -s "$TOPK_JSON" ]; then
        log "SKIP (already complete): ${CONDITION} seed=${SEED}"
        return 0
    fi
    # Clean slate: remove any stale artifacts from a previous partial/interrupted run.
    rm -f "$TRAINING_LOG" "$TOPK_JSON"
    rm -rf "$CKPT_DIR"
    log "============================================================"
    log "STARTING: ${CONDITION} seed=${SEED} (train: ${TRAIN_JSON})"
    log "============================================================"
    $PYTHON train.py \
        $COMMON_ARGS \
        --merdcir_json_path "${TRAIN_JSON}" \
        --topk_checkpoint_dir "${CKPT_DIR}" \
        --training_log_path "${TRAINING_LOG}" \
        --topk_json_path "${TOPK_JSON}" \
        --seed "${SEED}" \
        >> "$MAIN_LOG" 2>&1
    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: ${CONDITION} seed=${SEED} failed with exit code ${EXIT_CODE}"
        return $EXIT_CODE
    fi
    log "COMPLETED: ${CONDITION} seed=${SEED}"
}

mkdir -p ./checkpoints ./logs
log "============================================================"
log "ROUND-2 TRAINING START (remaining 5 conditions, priority order)"
log "============================================================"

run_training "MULTI" 42 "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_s42" \
    "./checkpoints/training_log_multi_s42.json" \
    "./checkpoints/topk_checkpoints_multi_s42.json"

run_training "FIXED" 42 "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_s42" \
    "./checkpoints/training_log_fixed_s42.json" \
    "./checkpoints/topk_checkpoints_fixed_s42.json"

run_training "RAW" 123 "pair_control/train.jsonl" \
    "./checkpoints/topk_pair_raw_s123" \
    "./checkpoints/training_log_raw_s123.json" \
    "./checkpoints/topk_checkpoints_raw_s123.json"

run_training "MULTI" 123 "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_s123" \
    "./checkpoints/training_log_multi_s123.json" \
    "./checkpoints/topk_checkpoints_multi_s123.json"

run_training "FIXED" 123 "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_s123" \
    "./checkpoints/training_log_fixed_s123.json" \
    "./checkpoints/topk_checkpoints_fixed_s123.json"

log "============================================================"
log "ROUND-2 TRAINING PIPELINE FINISHED"
log "============================================================"
