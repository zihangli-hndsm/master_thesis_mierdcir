#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zihali/final_thesis
PY=/home/zihali/data/conda/envs/thesis/bin/python
OUT="$ROOT/narrative_followup/predictions/cirr_val"
LOGDIR="$ROOT/narrative_followup/logs/cirr_val"
mkdir -p "$OUT" "$LOGDIR"

declare -A CKPT
CKPT[raw_42]=checkpoints/topk_pair_raw_s42/topk_epoch_0002_step_000600_score_0.432042.pth.tar
CKPT[raw_123]=checkpoints/topk_pair_raw_s123/topk_epoch_0002_step_000600_score_0.433568.pth.tar
CKPT[raw_2025]=checkpoints/topk_pair_raw_s2025/topk_epoch_0002_step_000600_score_0.434493.pth.tar
CKPT[fixed_42]=checkpoints/topk_pair_fixed_s42/topk_epoch_0002_step_000600_score_0.424210.pth.tar
CKPT[fixed_123]=checkpoints/topk_pair_fixed_s123/topk_epoch_0002_step_000600_score_0.426339.pth.tar
CKPT[fixed_2025]=checkpoints/topk_pair_fixed_s2025/topk_epoch_0002_step_000600_score_0.423249.pth.tar
CKPT[multi_42]=checkpoints/topk_pair_multi_s42/topk_epoch_0002_step_000600_score_0.441448.pth.tar
CKPT[multi_123]=checkpoints/topk_pair_multi_s123/topk_epoch_0002_step_000600_score_0.442584.pth.tar
CKPT[multi_2025]=checkpoints/topk_pair_multi_s2025/topk_epoch_0002_step_000600_score_0.440373.pth.tar

for cond in raw fixed multi; do
  for seed in 42 123 2025; do
    for metric in recall recall_subset; do
      out="$OUT/${cond}_s${seed}_${metric}.json"
      log="$LOGDIR/${cond}_s${seed}_${metric}.log"
      if [[ -s "$out" ]]; then
        echo "SKIP $out"
        continue
      fi
      echo "START condition=$cond seed=$seed metric=$metric"
      "$PY" "$ROOT/scripts/eval_checkpoint.py" \
        --checkpoint "$ROOT/${CKPT[${cond}_${seed}]}" \
        --dataset CIRR \
        --method merdcir_mlp_alpha \
        --backbone-size B \
        --cirr-metric "$metric" \
        --cirr-json-path cap.rc2.val.json \
        --cirr-split-json-path split.rc2.val.json \
        --cirr-force-export \
        --batch-size 256 \
        --output-json "$out" > "$log" 2>&1
      echo "DONE $out"
    done
  done
done
