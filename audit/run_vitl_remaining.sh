#!/bin/bash
# Remaining ViT-L pilot training (plan §8.3/§8.4), run in the NEXT session.
# Each item is skipped if its topk_json already exists (resume-friendly).
#
#  1. Multi-L s42        : train_multi.jsonl, ViT-L/14, batch 96, seed 42
#  2. Fixed-B' s42       : train_fixed.jsonl, ViT-B/32, batch 96 (batch-matched negatives, §8.4)
#  3. Multi-B' s42       : train_multi.jsonl, ViT-B/32, batch 96
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

run_training() {
    local TAG=$1 SEED=$2 BACKBONE=$3 BATCH=$4 TRAIN_JSON=$5 CKPT_DIR=$6 TRAINING_LOG=$7 TOPK_JSON=$8
    mkdir -p ./checkpoints ./logs
    local LOG="./logs/vitl_${TAG}.log"
    if [ -s "$TOPK_JSON" ]; then
        echo "[$(date '+%F %T')] SKIP (already complete): ${TAG}"
        return 0
    fi
    echo "[$(date '+%F %T')] START: ${TAG} backbone=${BACKBONE} batch=${BATCH} seed=${SEED}"
    $PYTHON scripts/train.py \
        --method merdcir_mlp_alpha \
        --lmdb_path ./data/MTCIR/images_224_lmdb \
        --eval_json_path pair_control/dev.jsonl \
        --resume_path ./checkpoints/_no_resume.pth.tar \
        --epochs 3 \
        --batch_size "${BATCH}" \
        --backbone_size "${BACKBONE}" \
        --skip_cirr_validation \
        --merdcir_json_path "${TRAIN_JSON}" \
        --topk_checkpoint_dir "${CKPT_DIR}" \
        --training_log_path "${TRAINING_LOG}" \
        --topk_json_path "${TOPK_JSON}" \
        --seed "${SEED}" \
        >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] EXIT $? : ${TAG}"
}

run_training "multi_L_s42" 42 L 96 "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_L_s42" \
    "./checkpoints/training_log_multi_L_s42.json" \
    "./checkpoints/topk_checkpoints_multi_L_s42.json"

run_training "fixed_B96_s42" 42 B 96 "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_B96_s42" \
    "./checkpoints/training_log_fixed_B96_s42.json" \
    "./checkpoints/topk_checkpoints_fixed_B96_s42.json"

run_training "multi_B96_s42" 42 B 96 "pair_control/train_multi.jsonl" \
    "./checkpoints/topk_pair_multi_B96_s42" \
    "./checkpoints/training_log_multi_B96_s42.json" \
    "./checkpoints/topk_checkpoints_multi_B96_s42.json"

echo "[$(date '+%F %T')] ALL REMAINING TRAINING DONE"
