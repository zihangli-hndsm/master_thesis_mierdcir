#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zihali/final_thesis
PY=/home/zihali/data/conda/envs/thesis/bin/python
OUT="$ROOT/narrative_followup/metrics"
LOGDIR="$ROOT/narrative_followup/logs/query_ranks"
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

declare -A DATA
DATA[raw]=pair_control/test.jsonl
DATA[fixed]=pair_control/test_fixed.jsonl
DATA[multi]=pair_control/test_multi.jsonl

for cond in raw fixed multi; do
  for seed in 42 123 2025; do
    for style in raw fixed multi; do
      out="$OUT/query_ranks_${cond}_s${seed}_${style}.csv"
      log="$LOGDIR/query_ranks_${cond}_s${seed}_${style}.log"
      if [[ -s "$out" ]] && [[ "$(wc -l < "$out")" -eq 10001 ]]; then
        echo "SKIP $out"
        continue
      fi
      echo "START condition=$cond seed=$seed query_style=$style"
      "$PY" "$ROOT/scripts/eval_checkpoint.py" \
        --checkpoint "$ROOT/${CKPT[${cond}_${seed}]}" \
        --dataset MTCIR \
        --method merdcir_mlp_alpha \
        --backbone-size B \
        --eval-json-path "${DATA[$style]}" \
        --exact-gallery \
        --batch-size 256 \
        --ranks-output "$out" > "$log" 2>&1
      echo "DONE $out"
    done
  done
done
