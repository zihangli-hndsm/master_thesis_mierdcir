# AGENTS.md — ScheiCIR Project Agent Guide

## Overview

This is a Composed Image Retrieval (CIR) thesis project. Given a reference image and a short modification text (e.g. "make the background darker"), the model retrieves the matching target image from a gallery.

The core contribution is **ScheiCIR**, a CLIP-based model that uses cross-attention between image patches and text tokens, plus a noun-phrase-level soft contrastive loss to learn fine-grained visual modifications.

## Project Layout

```text
final_thesis/
├── scripts/                          # Entry-point scripts and launchers
│   ├── train.py                      # Main training loop (3-stage warmup + joint)
│   ├── eval_checkpoint.py            # Single-checkpoint evaluation on CIR datasets
│   ├── evaluate_topk_checkpoints.py  # Batch eval: every .pth.tar in checkpoints/topk*/
│   ├── eval.py                       # ScheiEvaluator helper (lightweight ChromaDB wrapper)
│   ├── visualize_attention.py        # NP-level spatial attention map overlay
│   ├── app.py                        # Gradio human-evaluation app
│   ├── analyze_human_rating_results.py  # CSV→PDF analysis for human annotations
│   ├── gen_np.py                     # Extract noun phrases via benepar+spaCy→CLIP spans
│   ├── split_for.py                  # Split a large JSONL into N shards
│   └── run_all.sh                    # Launch gen_np.py on 12 JSONL shards in parallel
├── docs/                             # Notes, guidelines, and experiment reports
├── models/
│   ├── full_model.py                 # VisionEncoder, TextEncoder, AttentionPooler, ScheiCIR
│   └── interaction.py                # CrossAttentionBlock, AlphaGenerator, TransformerAlphaGenerator
├── data/
│   ├── dataset.py                    # MTCIRDataset, MerdCIRDataset, LaSCoDataset
│   ├── rewrite_pipeline.py           # vLLM pipeline: MTCIR→MerdCIR rewrite
│   ├── merdcir_prompts.py            # 6 intent scenarios for VLM rewriting
│   ├── merdcir_utils.py              # Pragmatic randomness (noun/preposition dropping)
│   ├── transform.py                  # Data transforms
│   ├── split_merdcir_np.py           # Train/eval split for MerdCIR NP data
│   ├── merge_shuffle.sh              # Shell script: merge+shuffle JSONL shards
│   ├── sample_mierd.py               # Sampling utilities for MerdCIR
│   ├── CIRR/                         # CIRR dataset code (save_lmdb.py, README.md)
│   ├── FashionIQ/                    # FashionIQ dataset code
│   ├── MTCIR/                        # MTCIR dataset code (save_lmdb.py, README.md)
│   ├── LaSCo/                        # LaSCo dataset code (save_lmdb.py)
│   ├── mtcir_np/                     # NP-annotated MTCIR data (merge_jsonl.py)
│   └── merdcir_np/                   # NP-annotated MerdCIR data
├── checkpoints/                      # gitignored: model checkpoints and eval JSONs
├── chroma_db/                        # gitignored: persistent ChromaDB
├── logs/                             # gitignored: training/eval logs
├── plots/                            # gitignored: generated figures
├── human_annotation_images/          # gitignored
├── human_rating_results/             # gitignored: CSV, PDF, report outputs
└── reproduction_experiments/         # gitignored
```

## Key Concepts

### ScheiCIR Model (`models/full_model.py`)

```
Input: (reference_image, modification_text)
  → VisionEncoder (CLIP ViT): image → patch features [B, N_patches, D_vision]
  → TextEncoder (CLIP Text): text → token features [B, N_tokens, D_text]
  → N layers of CrossAttentionBlock: image patches attend to text tokens
  → AttentionPooler: aggregate patch features → [B, D_vision]
  → Output: fused query embedding [B, D_vision]
```

Three model variants controlled by `--method`:
- **`cross_attn_alpha`**: CrossAttentionBlock + MLP AlphaGenerator (default)
- **`cross_pooling_alpha`**: CrossAttentionBlock + TransformerAlphaGenerator (with cross-attn)
- **`cross_attn`**: CrossAttentionBlock only, no alpha generator

### AlphaGenerator (`models/interaction.py`)

The AlphaGenerator learns to weight noun-phrase (NP) pairs in the soft contrastive loss. It takes a pair of NP features, their attention maps, and the CLIP [CLS] embedding, and outputs a scalar weight. This prevents NP attention maps from collapsing (degenerating) across different NPs.

Two implementations:
- **`AlphaGenerator`**: MLP on concatenated [feat_1, feat_2, attn_1, attn_2, cls]
- **`TransformerAlphaGenerator`**: Cross-attention between NP pairs followed by an MLP head

### Training Stages (`train.py`)

Three-stage progressive unfreezing:

| Stage | global_step range | Unfrozen | LR |
|-------|------------------|----------|-----|
| `warmup_head` | 0 → joint_start_step | interaction, attn_pooler, alpha_gen | 1e-4 |
| `warmup_last_layer` | joint_start_step → 2×joint_start_step | + last encoder layer of both backbones | 1e-6 (backbone)|
| `joint` | 2×joint_start_step → end | all parameters (alpha_gen frozen if `--freeze_alpha_in_joint`) | 1e-6 (backbone)|

### Datasets

| Dataset | Source | What it is | Dataset class |
|---------|--------|-----------|---------------|
| MTCIR | External | Original composed image retrieval (image + modification text → target) | `MTCIRDataset` |
| MerdCIR | This project | MTCIR rewritten by VLM with 6 intent scenarios | `MerdCIRDataset` |
| CIRR | External | Composed Image Retrieval with Real-world images (rc2) | `EvalJsonDataset` |
| FashionIQ | External | Fashion product image retrieval | `FashionIQDataset` |
| LaSCo | External | Large-scale composed image retrieval | `LaSCoDataset` |

### Evaluation Metrics

- **Recall@K**: Fraction of queries where the target image appears in the top-K retrieved results
- **mAP**: Mean Average Precision (1/rank for single-target retrieval)
- **CIRR recall**: Global retrieval over the full gallery
- **CIRR recall_subset**: Retrieval constrained to the sample's candidate set (Recall@1,2,3)

## Environment & Dependencies

```text
Python 3.13.12
CUDA 13.0
GPU: NVIDIA A100

Core: torch, torchvision, transformers, accelerate
Vision: Pillow, opencv-python, matplotlib
Retrieval: chromadb, lmdb
NLP: spacy, benepar (for noun phrase extraction)
UI: gradio (for human evaluation app)
Optional: vllm (for VLM rewriting pipeline)
```

Activate conda env: `conda activate thesis` (from `/home/zihali/data/conda/envs/thesis`)

## Common Commands

### Setup

```bash
cd /home/zihali/final_thesis
conda activate thesis

# Download NLP models (needed for gen_np.py)
python -m spacy download en_core_web_md
python -m benepar.download benepar_en3
```

### Training

```bash
# MerdCIR MLP Alpha (most common)
python scripts/train.py \
  --method merdcir_mlp_alpha \
  --merdcir_json_path merdcir_np/test_train.jsonl \
  --lmdb_path ./data/MTCIR/images_224_lmdb \
  --cirr_data_path ./data/CIRR \
  --cirr_lmdb_path ./data/CIRR/images_224_lmdb \
  --topk_checkpoint_dir ./checkpoints/topk_merdcir_mlp

# All methods: mtcir_mlp_alpha, merdcir_no_alpha, merdcir_mlp_alpha,
#              merdcir_cross_attn_alpha, lasco_mlp_alpha
```

Key training arguments:
- `--joint_start_step 300`: controls when stages advance
- `--epochs 6`
- `--batch_size 300`
- `--sc_loss_lambda 30`: soft contrastive loss weight
- `--temperature 0.07`: InfoNCE temperature
- `--freeze_alpha_in_joint` / `--train_alpha_in_joint`
- `--resume_path`: path to checkpoint for resuming

### Evaluation

```bash
# Evaluate one checkpoint on a specific dataset:
python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/topk_merdcir_mlp/topk_epoch_0003_step_000900_score_0.483130.pth.tar \
  --dataset MTCIR \
  --method cross_attn_alpha

# Supported --dataset: MTCIR, MerdCIR, CIRR, FashionIQ

# CIRR requires --cirr-metric recall or recall_subset
python scripts/eval_checkpoint.py \
  --checkpoint <path> --dataset CIRR --method cross_attn_alpha \
  --cirr-metric recall --output-json checkpoints/topk/cirr_recall.json

# Batch evaluate all top-k folders:
python scripts/evaluate_topk_checkpoints.py --checkpoint-root checkpoints
```

After batch evaluation, the Markdown table is written to `docs/checkpoint_eval_results.md`.

### Noun Phrase Extraction

```bash
# Single file
python scripts/gen_np.py --input-jsonl data/MTCIR/mtcir.jsonl --output-jsonl output_nps.jsonl

# 12-way parallel on shards (edit run_all.sh to adjust paths)
bash scripts/run_all.sh
```

### VLM Rewriting (MTCIR → MerdCIR)

```bash
python data/rewrite_pipeline.py \
  --jsonl_path data/MTCIR/mtcir.jsonl \
  --lmdb_path data/MTCIR/images_224_lmdb \
  --output_path output_rewritten.jsonl \
  --model_name "Qwen/Qwen3.5-27B-FP8" \
  --batch_size 16
```

Uses vLLM with AWQ quantization. Requires `vllm` installed.

### Human Evaluation App

```bash
python scripts/app.py --samples samples.json --server-port 7860
# or with annotator preset:
python scripts/app.py --samples samples.json --annotator Annotator_1

# Shell wrappers:
bash scripts/run_annotator_1.sh
bash scripts/run_annotator_2.sh
```

The app:
1. Expects `samples.json` with 160 items (auto-generates mock data if missing).
2. Annotator_1 gets samples 0-79, Annotator_2 gets 80-159.
3. Saves results to `evaluation_results_Annotator_N.csv`.
4. Supports resume: reads existing CSV, continues from first unfinished sample.

### Analyze Human Ratings

```bash
python scripts/analyze_human_rating_results.py
```

Reads CSVs from `human_rating_results/`, produces:
- `human_rating_results/human_rating_analysis_report.md`
- `human_rating_results/human_rating_score_summary.csv`
- `human_rating_results/annotated_mierdcir_fail_cases.pdf`
- `human_rating_results/missing_annotation_entries.pdf`

### Attention Visualization

```bash
python scripts/visualize_attention.py \
  --id <SAMPLE_ID> \
  --checkpoint checkpoints/topk_merdcir_cross_attn/topk_epoch_0003_step_000900_score_0.483130.pth.tar \
  --text-source merdcir \
  --out-dir plots

# Options:
# --text-source: mtcir or merdcir
# --max-nps 10: max noun phrases to visualize
# --backbone-size B|L|H
# --allow-random-model: use untrained model if no checkpoint
```

Output: `plots/attention_fail_case_<ID>.pdf`

## Data Flow

```
External Datasets (CIRR, FashionIQ, MTCIR, LaSCo)
  ↓ (manual download, place under data/)
  ↓ save_lmdb.py → images_224_lmdb/
  ↓
MTCIR.jsonl
  ↓ scripts/rewrite_pipeline.py (VLM: Qwen3-27B)
MerdCIR (rewritten JSONL)
  ↓ scripts/gen_np.py (benepar + spaCy)
mtcir_np/ & merdcir_np/ (JSONL with NP spans)
  ↓ scripts/train.py
checkpoints/
  ↓ scripts/eval_checkpoint.py
  ↓ scripts/evaluate_topk_checkpoints.py
Metrics (JSON) + docs/checkpoint_eval_results.md
```

## JSONL Record Format

### MTCIR / MerdCIR training samples

```json
{
  "id": "unique_id",
  "image": "ref_image_key_for_lmdb",
  "target_img": "target_image_key_for_lmdb",
  "modification": ["sentence 1", "sentence 2"],
  "merdcir_modification": "rewritten concise query",
  "nps": [["noun phrase text", [start_token, end_token]], ...]
}
```

NP spans are 1-based inclusive CLIP token positions.

### CIRR

```json
{
  "pairid": 123,
  "reference": "test1/IMG_001",
  "target_hard": "test1/IMG_002",
  "caption": "modification text",
  "img_set": {"members": ["test1/IMG_001", ...]}
}
```

## Important Conventions

1. **Images are stored in LMDB**, not as files on disk. All dataset classes read via `lmdb.open(..., readonly=True, lock=False)`.
2. **NP spans are 1-based inclusive**. In `extract_np_attention_maps()`, the code does `start-1:end` to convert to 0-based Python slice.
3. **Checkpoints use `.pth.tar` extension**. The score is embedded in the filename: `topk_epoch_XXXX_step_YYYYYY_score_0.ZZZZZZ.pth.tar`.
4. **Eval uses ChromaDB** for vector similarity search. Collections are created per evaluation run and deleted after.
5. **CIRR evaluation has two modes**: global `recall` (full gallery) and `recall_subset` (within candidate set).
6. **Training uses 8 DataLoader workers with `pin_memory=True`**; validation uses 0 workers to avoid LMDB forking issues.
7. **All large artifacts are gitignored**: checkpoints, LMDB, images, JSONL outputs, logs, plots, PDFs.
8. **The conda environment** is at `/home/zihali/data/conda/envs/thesis` (Python 3.13).

## Top-k Checkpoint Results (Baseline)

From `docs/checkpoint_eval_results.md`, best MerdCIR results on the evaluation set:

| Method | MTCIR R@1 | MTCIR R@10 | MerdCIR R@1 | FashionIQ R@1 |
|--------|-----------|------------|-------------|---------------|
| topk_merdcir_cross_attn (cross_pooling_alpha) | 0.4865 | 0.8284 | 0.7208 | 0.0609 |
| topk_new_dataset (cross_attn_alpha) | 0.4901 | 0.8276 | 0.7162 | 0.0625 |
| topk_baseline (cross_attn) | 0.4811 | 0.8286 | 0.7232 | 0.0603 |

## Troubleshooting

- **LMDB lock errors in validation**: Use `num_workers=0` for validation DataLoaders (LMDB doesn't support fork-safe multi-processing).
- **"No module named benepar"**: Install with `pip install benepar` and download the model.
- **CUDA OOM during training**: Reduce `--batch_size`. The default 300 is tuned for A100 80GB.
- **CIRR JSON appears as HTML**: The file was downloaded as an HTML page from the hosting service. Re-download as raw JSON.
- **Missing images in LMDB**: Run the appropriate `save_lmdb.py` script in each dataset directory.
