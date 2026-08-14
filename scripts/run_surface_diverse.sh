#!/bin/bash
# Surface-Diverse control (exploratory, per FOLLOWUP_EXPERIMENT_PLAN.md §4.2 + §6.2):
#   1) Rewrite a 50K slice of the frozen pair_control train set with 6 surface templates
#      expressing the SAME generic retrieval intent (wording-only diversity).
#   2) Extract noun phrases (gen_np.py).
#   3) Train 1 seed (exploratory) on the Surface-Diverse corpus.
# Run AFTER RAW s2025 training finishes (GPU is needed for vLLM generation).
# Usage: bash run_surface_diverse.sh

set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

N_SAMPLES=${1:-50000}
SEED=${2:-2026}
OUT_DIR="$BASE_DIR/data/pair_control"
RAW_OUT="$OUT_DIR/train_surfacediverse_raw.jsonl"
NP_OUT="$OUT_DIR/train_surfacediverse.jsonl"
CKPT_DIR="./checkpoints/topk_pair_sd_s${SEED}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="./logs/surface_diverse_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

log "============================================================"
log "SURFACE-DIVERSE PIPELINE START (N=${N_SAMPLES}, seed=${SEED})"
log "============================================================"

# Step 1: Rewrite with 6 surface templates (vLLM, GPU)
log "STEP 1: Surface-diverse rewriting (N=${N_SAMPLES})"
$PYTHON "$BASE_DIR/data/rewrite_surface_diverse.py" \
    --input-jsonl "$OUT_DIR/train.jsonl" \
    --output-jsonl "$RAW_OUT" \
    --model-name "Qwen/Qwen3.5-27B-FP8" \
    --batch-size 64 \
    --num-workers 4 \
    --max-samples "$N_SAMPLES" \
    2>&1 | tee -a "$MAIN_LOG"

# Step 2: Extract noun phrases
log "STEP 2: Noun-phrase extraction"
$PYTHON "$BASE_DIR/gen_np.py" \
    --input-jsonl "$RAW_OUT" \
    --output-jsonl "$NP_OUT" \
    --batch-size 2048 \
    2>&1 | tee -a "$MAIN_LOG"

# Step 3: Exploratory training (1 seed) on Surface-Diverse corpus
log "STEP 3: Training (exploratory, seed=${SEED})"
$PYTHON scripts/train.py \
    --method merdcir_mlp_alpha \
    --lmdb_path ./data/MTCIR/images_224_lmdb \
    --cirr_data_path ./data/CIRR \
    --cirr_json_path cap.rc2.val.json \
    --cirr_split_json_path split.rc2.val.json \
    --cirr_lmdb_path ./data/CIRR/images_224_lmdb \
    --eval_json_path pair_control/dev.jsonl \
    --resume_path ./checkpoints/_no_resume.pth.tar \
    --epochs 3 \
    --batch_size 300 \
    --merdcir_json_path pair_control/train_surfacediverse.jsonl \
    --topk_checkpoint_dir "$CKPT_DIR" \
    --training_log_path "./checkpoints/training_log_sd_s${SEED}.json" \
    --topk_json_path "./checkpoints/topk_checkpoints_sd_s${SEED}.json" \
    --seed "$SEED" \
    2>&1 | tee -a "$MAIN_LOG"

log "============================================================"
log "SURFACE-DIVERSE PIPELINE FINISHED"
log "============================================================"
