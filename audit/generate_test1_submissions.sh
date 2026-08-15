#!/bin/bash
# Generate CIRR test1 Recall + RecallSubset submission JSONs for the frozen
# ViT-B checkpoint set (9 models), validate each, and record hashes.
# ViT-L additions (if the §8.7 trigger passes) go into the *_L_* entries below.
# Usage: bash audit/generate_test1_submissions.sh
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./cirr_test_results/test1_gen_${TS}.log"
mkdir -p ./cirr_test_results
echo "[$(date '+%F %T')] test1 generation start" | tee "$LOG"

# condition|seed|backbone|checkpoint  (ViT-B set from the frozen manifest)
MODELS=(
  "raw|s42|B|checkpoints/topk_pair_raw_s42/topk_epoch_0002_step_000600_score_0.432042.pth.tar"
  "raw|s123|B|checkpoints/topk_pair_raw_s123/topk_epoch_0002_step_000600_score_0.433568.pth.tar"
  "raw|s2025|B|checkpoints/topk_pair_raw_s2025/topk_epoch_0002_step_000600_score_0.434493.pth.tar"
  "fixed|s42|B|checkpoints/topk_pair_fixed_s42/topk_epoch_0002_step_000600_score_0.424210.pth.tar"
  "fixed|s123|B|checkpoints/topk_pair_fixed_s123/topk_epoch_0002_step_000600_score_0.426339.pth.tar"
  "fixed|s2025|B|checkpoints/topk_pair_fixed_s2025/topk_epoch_0002_step_000600_score_0.423249.pth.tar"
  "multi|s42|B|checkpoints/topk_pair_multi_s42/topk_epoch_0002_step_000600_score_0.441448.pth.tar"
  "multi|s123|B|checkpoints/topk_pair_multi_s123/topk_epoch_0002_step_000600_score_0.442584.pth.tar"
  "multi|s2025|B|checkpoints/topk_pair_multi_s2025/topk_epoch_0002_step_000600_score_0.440373.pth.tar"
)
# ViT-L additions go here once the pilot decision is made (backbone L):
# "fixed|s42|L|checkpoints/topk_pair_fixed_L_s42/step2400.pth.tar"
# "multi|s42|L|checkpoints/topk_pair_multi_L_s42/<best>.pth.tar"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r COND SEED BB CKPT <<< "$entry"
    if [ ! -f "$CKPT" ]; then
        echo "[$(date '+%F %T')] SKIP missing ckpt: $CKPT" | tee -a "$LOG"
        continue
    fi
    echo "[$(date '+%F %T')] ${COND}_${SEED} (backbone=$BB): generating recall" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset CIRR --method cross_attn_alpha --backbone-size "$BB" \
        --cirr-metric recall \
        --output-json "cirr_test_results/${COND}_${SEED}_recall.json" \
        >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] ${COND}_${SEED}: generating recall_subset" | tee -a "$LOG"
    $PYTHON scripts/eval_checkpoint.py \
        --checkpoint "$CKPT" --dataset CIRR --method cross_attn_alpha --backbone-size "$BB" \
        --cirr-metric recall_subset \
        --output-json "cirr_test_results/${COND}_${SEED}_recall_subset.json" \
        >> "$LOG" 2>&1
    for METRIC in recall recall_subset; do
        F="cirr_test_results/${COND}_${SEED}_${METRIC}.json"
        $PYTHON audit/cirr_validate.py --pred "$F" \
            --gt data/CIRR/cap.rc2.test1.json \
            --split data/CIRR/split.rc2.test1.json \
            --checkpoint "$CKPT" >> "$LOG" 2>&1
        echo "[$(date '+%F %T')] validated: $F" | tee -a "$LOG"
    done
done
echo "[$(date '+%F %T')] test1 generation complete" | tee -a "$LOG"
