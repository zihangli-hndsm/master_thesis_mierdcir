#!/bin/bash
# ViT-L capacity pilot stage 3 (§8.4): ViT-B batch-matched re-runs (batch 96, 1 epoch).
# Fixed-B' / Multi-B' s42 — same per-forward batch and step budget as the ViT-L runs,
# so in-batch negative counts and update counts are matched across backbones.
# Reduced protocol (2026-08-14 decision): matched-step 2400 (epoch 0) instead of 3 epochs.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

BATCH=96
BACKBONE=B

COMMON_ARGS="
  --method merdcir_mlp_alpha
  --lmdb_path ./data/MTCIR/images_224_lmdb
  --eval_json_path pair_control/dev.jsonl
  --resume_path ./checkpoints/_no_resume.pth.tar
  --epochs 1
  --batch_size ${BATCH}
  --backbone_size ${BACKBONE}
  --skip_cirr_validation
"

run_training() {
    local TAG=$1 SEED=$2 TRAIN_JSON=$3 CKPT_DIR=$4 TRAINING_LOG=$5 TOPK_JSON=$6
    mkdir -p ./checkpoints ./logs
    local LOG="./logs/vitl_${TAG}_s${SEED}.log"
    if [ -s "$TOPK_JSON" ]; then
        echo "[$(date '+%F %T')] SKIP (already complete): ${TAG} seed=${SEED}"
        return 0
    fi
    echo "[$(date '+%F %T')] START: ${TAG} seed=${SEED} backbone=B batch=96"
    $PYTHON scripts/train.py \
        $COMMON_ARGS \
        --merdcir_json_path "${TRAIN_JSON}" \
        --topk_checkpoint_dir "${CKPT_DIR}" \
        --training_log_path "${TRAINING_LOG}" \
        --topk_json_path "${TOPK_JSON}" \
        --seed "${SEED}" \
        >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] EXIT $? : ${TAG} seed=${SEED}"
}

run_training "fixed_B96" 42 "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_B96_s42" \
    "./checkpoints/training_log_fixed_B96_s42.json" \
    "./checkpoints/topk_checkpoints_fixed_B96_s42.json"

run_training "multi_B96" 42 "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_B96_s42" \
    "./checkpoints/training_log_multi_B96_s42.json" \
    "./checkpoints/topk_checkpoints_multi_B96_s42.json"
