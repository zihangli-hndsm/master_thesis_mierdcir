#!/usr/bin/env bash
# Train the strict surface-only control on the frozen 255.4K Fixed universe.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd "$BASE_DIR"

for seed in 42 123 2025; do
  tag="surface_v2_s${seed}"
  ckpt_dir="./checkpoints/topk_${tag}"
  train_log="./checkpoints/training_log_${tag}.json"
  topk_json="./checkpoints/topk_checkpoints_${tag}.json"
  log="./logs/${tag}.log"
  if [ -s "$topk_json" ]; then
    echo "[$(date '+%F %T')] SKIP complete: $tag"
    continue
  fi
  mkdir -p ./checkpoints ./logs
  echo "[$(date '+%F %T')] START: $tag" | tee "$log"
  "$PYTHON" scripts/train.py \
    --method merdcir_mlp_alpha \
    --lmdb_path ./data/MTCIR/images_224_lmdb \
    --eval_json_path pair_control/dev.jsonl \
    --resume_path ./checkpoints/_no_resume.pth.tar \
    --epochs 3 --batch_size 300 --skip_cirr_validation \
    --merdcir_json_path pair_control_surface_v2/train_np.jsonl \
    --topk_checkpoint_dir "$ckpt_dir" \
    --training_log_path "$train_log" \
    --topk_json_path "$topk_json" --seed "$seed" \
    >> "$log" 2>&1
  status=$?
  echo "[$(date '+%F %T')] EXIT $status: $tag" | tee -a "$log"
  [ "$status" -eq 0 ] || exit "$status"
done
