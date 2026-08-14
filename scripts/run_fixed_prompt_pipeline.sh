#!/bin/bash
# Full pipeline: rewrite part_0 with fixed prompt → extract NPs → train/eval split
# Usage: bash run_fixed_prompt_pipeline.sh

set -e
PROJ="/home/zihali/final_thesis"
PY="/home/zihali/data/conda/envs/thesis/bin/python"
PART="part_0"
OUT_DIR="$PROJ/data/fixed_prompt"

mkdir -p "$OUT_DIR"
mkdir -p "$PROJ/data/fixed_prompt"

echo "=== Step 1: Rewrite with fixed prompt (no intent diversity) ==="
$PY "$PROJ/data/rewrite_single_prompt.py" \
    --input-jsonl "$PROJ/$PART.jsonl" \
    --output-jsonl "$OUT_DIR/${PART}_rewritten.jsonl" \
    --model-name "Qwen/Qwen3.5-27B-FP8" \
    --batch-size 64 \
    --num-workers 4 \
    2>&1 | tee "$OUT_DIR/rewrite.log"

echo "=== Step 2: Extract noun phrases ==="
$PY "$PROJ/scripts/gen_np.py" \
    --input-jsonl "$OUT_DIR/${PART}_rewritten.jsonl" \
    --output-jsonl "$OUT_DIR/${PART}_nps.jsonl" \
    --batch-size 2048 \
    2>&1 | tee "$OUT_DIR/gen_np.log"

echo "=== Step 3: Train/eval split ==="
$PY "$PROJ/data/split_merdcir_np.py" \
    --input-jsonl "$OUT_DIR/${PART}_nps.jsonl" \
    --output-dir "$OUT_DIR" \
    --eval-size 5000 \
    --seed 114514 \
    2>&1 | tee "$OUT_DIR/split.log"

echo "=== Done ==="
echo "Train file: $OUT_DIR/train.jsonl"
echo "Eval file:  $OUT_DIR/eval.jsonl"
echo "Full output: $OUT_DIR/${PART}_nps.jsonl"
ls -lh "$OUT_DIR/"
