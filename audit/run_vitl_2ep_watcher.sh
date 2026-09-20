#!/bin/bash
# Watcher: waits for the Fixed-L 2ep run to finish, then auto-starts the
# Multi-L 2ep resume. Survives across loop sessions (runs under setsid).
# Usage: setsid nohup bash audit/run_vitl_2ep_watcher.sh <fixed_L_pid> > logs/vitl_2ep_watcher.log 2>&1 < /dev/null &

set -u
FIXED_PID=$1
PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

LOG="./logs/vitl_2ep_watcher.log"
echo "[$(date '+%F %T')] watcher started, waiting for Fixed-L (pid $FIXED_PID)" >> "$LOG"

while kill -0 "$FIXED_PID" 2>/dev/null; do
    sleep 120
done
echo "[$(date '+%F %T')] Fixed-L finished (pid $FIXED_PID gone)" >> "$LOG"

# safety: wait for GPU to free
sleep 60

# Confirm no residual training process before launching Multi-L
if ps -eo cmd | grep -q "scripts/train.py"; then
    echo "[$(date '+%F %T')] ERROR: residual train.py running, NOT auto-launching Multi-L" >> "$LOG"
    exit 1
fi

echo "[$(date '+%F %T')] launching Multi-L 2ep resume" >> "$LOG"
setsid nohup "$PYTHON" -u scripts/train.py \
    --method merdcir_mlp_alpha \
    --lmdb_path ./data/MTCIR/images_224_lmdb \
    --eval_json_path pair_control/dev.jsonl \
    --resume_path ./checkpoints/topk_pair_multi_L_s42/topk_epoch_0000_step_002660_score_0.118471.pth.tar \
    --epochs 2 \
    --batch_size 96 \
    --backbone_size L \
    --skip_cirr_validation \
    --val_interval 600 \
    --merdcir_json_path pair_control/train_multi.jsonl \
    --topk_checkpoint_dir ./checkpoints/topk_pair_multi_L_s42_2ep \
    --training_log_path ./checkpoints/training_log_multi_L_s42_2ep.json \
    --topk_json_path ./checkpoints/topk_checkpoints_multi_L_s42_2ep.json \
    --seed 42 \
    > ./logs/vitl_multi_L_2ep_s42.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] Multi-L 2ep launched (pid $!)" >> "$LOG"
