#!/bin/bash
# CIRR val rehearsal (plan §5): exporter consistency + batch-size invariance + order invariance.
# Uses one frozen checkpoint (multi s42). Requires GPU; run after MTCIR GPU audit.
set -u

PYTHON=/home/zihali/data/conda/envs/thesis/bin/python3
BASE_DIR=/home/zihali/final_thesis
cd $BASE_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="./logs/cirr_val_rehearsal_${TS}.log"
mkdir -p ./logs ./cirr_test_results/val_rehearsal

CKPT="./checkpoints/topk_pair_multi_s42/topk_epoch_0002_step_000600_score_0.441448.pth.tar"
METHOD="cross_attn_alpha"
OUTDIR="./cirr_test_results/val_rehearsal"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== CIRR val rehearsal start (checkpoint: $(basename $CKPT)) ==="

# 1. Direct local evaluation (reference numbers, §5 comparison target)
log "--- direct recall eval (batch 128) ---"
$PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
    --cirr-metric recall --cirr-json-path cap.rc2.val.json 2>&1 | tee -a "$LOG"
log "--- direct subset eval (batch 128) ---"
$PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
    --cirr-metric recall_subset --cirr-json-path cap.rc2.val.json 2>&1 | tee -a "$LOG"

# 2. Exporter on val with labels (force export), batch 128
log "--- export val recall (batch 128) ---"
$PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
    --cirr-metric recall --cirr-force-export --cirr-json-path cap.rc2.val.json \
    --output-json "$OUTDIR/val_rehearsal_recall_bs128.json" 2>&1 | tee -a "$LOG"
log "--- export val recall_subset (batch 128) ---"
$PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
    --cirr-metric recall_subset --cirr-force-export --cirr-json-path cap.rc2.val.json \
    --output-json "$OUTDIR/val_rehearsal_recall_subset_bs128.json" 2>&1 | tee -a "$LOG"

# 3. Batch-size invariance: re-export with batch 64
log "--- export val recall (batch 64) ---"
$PYTHON eval_checkpoint.py --checkpoint "$CKPT" --dataset CIRR --method "$METHOD" \
    --cirr-metric recall --cirr-force-export --cirr-json-path cap.rc2.val.json --batch-size 64 \
    --output-json "$OUTDIR/val_rehearsal_recall_bs64.json" 2>&1 | tee -a "$LOG"

# 4. Recompute metrics from exported JSONs with target_hard GT
log "--- recompute from exported JSONs ---"
$PYTHON compute_cirr_metrics.py --gt data/CIRR/cap.rc2.val.json \
    --pred-dir "$OUTDIR" 2>&1 | tee -a "$LOG"

log "=== CIRR val rehearsal complete ==="
