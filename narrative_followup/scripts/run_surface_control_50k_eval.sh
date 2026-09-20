#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zihali/final_thesis
PY=/home/zihali/data/conda/envs/thesis/bin/python
cd "$ROOT"
OUT="$ROOT/narrative_followup/metrics"
LOGDIR="$ROOT/narrative_followup/logs"
mkdir -p "$OUT" "$LOGDIR"

checkpoint_for() {
  local condition="$1" seed="$2"
  "$PY" - "$condition" "$seed" <<'PY'
import json, pathlib, sys
condition, seed = sys.argv[1:]
p = pathlib.Path(f"checkpoints/topk_checkpoints_surface50k_{condition}_s{seed}.json")
if not p.is_file():
    raise SystemExit(2)
rows = json.loads(p.read_text())
if not rows:
    raise SystemExit(3)
print(pathlib.Path(rows[0]["path"]).resolve())
PY
}

declare -A DATA
DATA[raw]=pair_control/test.jsonl
DATA[fixed]=pair_control/test_fixed.jsonl
DATA[multi]=pair_control/test_multi.jsonl
DATA[surface]=pair_control_surface_v2/test.jsonl

for condition in fixed surface multi; do
  for seed in 42 123 2025; do
    ckpt="$(checkpoint_for "$condition" "$seed")" || {
      echo "SKIP missing checkpoint: ${condition}_s${seed}"
      continue
    }
    for style in raw fixed multi surface; do
      out="$OUT/surface50k_query_${condition}_s${seed}_${style}.json"
      log="$LOGDIR/surface50k_query_${condition}_s${seed}_${style}.log"
      if [[ -s "$out" ]] && grep -q '^mAP:' "$log"; then
        echo "SKIP $out"
        continue
      fi
      echo "START condition=$condition seed=$seed query_style=$style"
      "$PY" scripts/eval_checkpoint.py \
        --checkpoint "$ckpt" \
        --dataset MTCIR \
        --method merdcir_mlp_alpha \
        --backbone-size B \
        --eval-json-path "${DATA[$style]}" \
        --exact-gallery \
        --batch-size 256 \
        --output-json "$out" > "$log" 2>&1
      "$PY" narrative_followup/scripts/parse_eval_log.py "$log" "$out"
      echo "DONE $out"
    done
    for metric in recall recall_subset; do
      out="$OUT/surface50k_cirr_${condition}_s${seed}_${metric}.json"
      log="$LOGDIR/surface50k_cirr_${condition}_s${seed}_${metric}.log"
      if [[ -s "$out" ]]; then
        echo "SKIP $out"
        continue
      fi
      echo "START CIRR condition=$condition seed=$seed metric=$metric"
      "$PY" scripts/eval_checkpoint.py \
        --checkpoint "$ckpt" \
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
