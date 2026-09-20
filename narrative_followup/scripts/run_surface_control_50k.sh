#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zihali/final_thesis
PY=/home/zihali/data/conda/envs/thesis/bin/python
cd "$ROOT"

for seed in 42 123 2025; do
  for condition in fixed surface multi; do
    tag="surface50k_${condition}_s${seed}"
    ckpt_dir="./checkpoints/topk_${tag}"
    train_log="./checkpoints/training_log_${tag}.json"
    topk_json="./checkpoints/topk_checkpoints_${tag}.json"
    log="./narrative_followup/logs/${tag}.log"
    data_file="pair_control_surface_50k/train_${condition}.jsonl"
    if [[ -s "$topk_json" ]]; then
      echo "[$(date '+%F %T')] SKIP complete: $tag"
      continue
    fi
    mkdir -p ./checkpoints ./narrative_followup/logs
    echo "[$(date '+%F %T')] START: $tag" | tee "$log"
    "$PY" scripts/train.py \
      --method merdcir_mlp_alpha \
      --lmdb_path ./data/MTCIR/images_224_lmdb \
      --eval_json_path pair_control/dev.jsonl \
      --resume_path ./checkpoints/_no_resume.pth.tar \
      --epochs 3 --batch_size 300 --skip_cirr_validation \
      --merdcir_json_path "$data_file" \
      --topk_checkpoint_dir "$ckpt_dir" \
      --training_log_path "$train_log" \
      --topk_json_path "$topk_json" --seed "$seed" \
      >> "$log" 2>&1
    echo "[$(date '+%F %T')] DONE: $tag" | tee -a "$log"
  done
done
