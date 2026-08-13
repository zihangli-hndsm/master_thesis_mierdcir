#!/bin/bash
# ViT-L capacity pilot (plan §8.3): single-seed 2x2 with batch-matched negatives (§8.4).
# Stage 1 (this session): Fixed-L s42  (starts now, resumes across sessions via checkpointing)
# Stage 2 (next session): Multi-L s42
# Stage 3 (next session): ViT-B batch-matched re-runs Fixed-B' / Multi-B' s42
#
# Protocol identical to pair_control: merdcir_mlp_alpha, 3 epochs, common-dev
# (pair_control/dev.jsonl) checkpoint selection, seed 42. CIRR val validation is
# skipped during training to save wall-clock; CIRR numbers are evaluated after.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

BATCH=${1:-96}   # per-forward batch (probe: max stable 128 w/ simple loss; real step OK at 96 = 61GB peak)
BACKBONE=L

COMMON_ARGS="
  --method merdcir_mlp_alpha
  --lmdb_path ./data/MTCIR/images_224_lmdb
  --eval_json_path pair_control/dev.jsonl
  --resume_path ./checkpoints/_no_resume.pth.tar
  --epochs 3
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
    echo "[$(date '+%F %T')] START: ${TAG} seed=${SEED} backbone=${BACKBONE} batch=${BATCH}"
    $PYTHON train.py \
        $COMMON_ARGS \
        --merdcir_json_path "${TRAIN_JSON}" \
        --topk_checkpoint_dir "${CKPT_DIR}" \
        --training_log_path "${TRAINING_LOG}" \
        --topk_json_path "${TOPK_JSON}" \
        --seed "${SEED}" \
        >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] EXIT $? : ${TAG} seed=${SEED}"
}

run_training "fixed" 42 "pair_control/train_fixed.jsonl" \
    "./checkpoints/topk_pair_fixed_L_s42" \
    "./checkpoints/training_log_fixed_L_s42.json" \
    "./checkpoints/topk_checkpoints_fixed_L_s42.json"
