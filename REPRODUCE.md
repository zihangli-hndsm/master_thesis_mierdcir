# Reproduction Guide

This guide documents the expected local setup for reproducing the thesis experiments. Large datasets, LMDB stores, checkpoints, generated JSON/JSONL files, plots, and logs are intentionally not committed to git.

Before restoring datasets, read `DATASET_NOTICE.md`. Third-party datasets are not redistributed by this repository. Generated data created by this project is intended to be released under CC BY 4.0 where permitted, but users must still comply with source dataset terms.

## 1. Environment

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training/evaluation, install a PyTorch build that matches your CUDA driver before installing the rest of the dependencies. If you use the VLM rewrite pipeline in `data/rewrite_pipeline.py`, also install `vllm` in a CUDA-capable Linux environment.

Some noun-phrase utilities require external language models:

```bash
python -m spacy download en_core_web_md
python -m benepar.download benepar_en3
```

Tested environment:

```text
Python 3.13.12
CUDA 13.0
GPU: NVIDIA A100
```

## 2. Restore Datasets

Place datasets under `data/` using this structure:

```text
data/
├── CIRR/
│   ├── cap.rc2.val.json
│   ├── cap.rc2.test1.json
│   ├── split.rc2.val.json
│   ├── split.rc2.test1.json
│   └── img_raw_filtered/
├── FashionIQ/
│   ├── captions/
│   ├── image_splits/
│   └── images/asin2url.{dress,shirt,toptee}.txt
├── MTCIR/
│   ├── mtcir.jsonl
│   └── images/
└── LaSCo/
    ├── captions/
    └── images/
```

Dataset licenses and download permissions are handled outside this repository. Training images come from the LLaVA-pretrain dataset available on Hugging Face, which is a subset of Google's CC3M. These images may include harmful or sensitive content, so apply appropriate safety review and filtering.

## 3. Build LMDB Image Stores

Build CIRR:

```bash
cd data/CIRR
python save_lmdb.py
cd ../..
```

Build MTCIR:

```bash
cd data/MTCIR
python save_lmdb.py
cd ../..
```

Build FashionIQ validation and test LMDBs:

```bash
cd data/FashionIQ
python save_lmdb.py --data-root . --split val
python save_lmdb.py --data-root . --split test
cd ../..
```

Build LaSCo:

```bash
python data/LaSCo/save_lmdb.py \
  --image-root data/LaSCo/images \
  --output-lmdb data/LaSCo/images_224_lmdb
```

The generated `images_224_lmdb/`, `images_val_224_lmdb/`, and `images_test_224_lmdb/` folders are ignored by git.

## 4. Prepare JSONL Files

Generate noun-phrase annotations when needed:

```bash
python gen_np.py \
  --input-jsonl data/MTCIR/mtcir.jsonl \
  --output-jsonl output_nps.jsonl
```

Split a large JSONL into shards:

```bash
python split_for.py \
  --input-jsonl data/rewrite.jsonl \
  --num-splits 12 \
  --output-dir .
```

Merge local JSONL shards:

```bash
cd data/mtcir_np
python merge_jsonl.py --pattern 'part_*.jsonl' --output-jsonl merged.jsonl
cd ../..
```

Create a reproducible MerdCIR train/eval split:

```bash
python data/split_merdcir_np.py \
  --input-jsonl data/merdcir_np/merged.jsonl \
  --output-dir data/merdcir_np \
  --eval-size 5000 \
  --seed 114514
```

## 5. Train

Example MerdCIR training command:

```bash
python train.py \
  --method merdcir_mlp_alpha \
  --merdcir_json_path merdcir_np/test_train.jsonl \
  --lmdb_path ./data/MTCIR/images_224_lmdb \
  --cirr_data_path ./data/CIRR \
  --cirr_json_path cap.rc2.val.json \
  --cirr_split_json_path split.rc2.val.json \
  --cirr_lmdb_path ./data/CIRR/images_224_lmdb \
  --topk_checkpoint_dir ./checkpoints/topk_merdcir_mlp
```

Checkpoints and training logs are generated under ignored local directories.

## 6. Evaluate

Evaluate a checkpoint on MTCIR:

```bash
python eval_checkpoint.py \
  --checkpoint checkpoints/topk_merdcir_mlp/example.pth.tar \
  --dataset MTCIR \
  --method cross_attn_alpha \
  --output-json checkpoints/topk_merdcir_mlp/mtcir_metrics.json
```

Evaluate on CIRR:

```bash
python eval_checkpoint.py \
  --checkpoint checkpoints/topk_merdcir_mlp/example.pth.tar \
  --dataset CIRR \
  --method cross_attn_alpha \
  --cirr-metric recall \
  --output-json checkpoints/topk_merdcir_mlp/cirr_recall.json
```

Evaluate all top-k checkpoint folders:

```bash
python evaluate_topk_checkpoints.py --checkpoint-root checkpoints
```

## 7. Human Evaluation And Visualization

Run the human-evaluation app:

```bash
python app.py --samples samples.json --server-port 7860
```

Generate an attention visualization:

```bash
python visualize_attention.py \
  --samples samples.json \
  --checkpoint checkpoints/topk_merdcir_mlp/example.pth.tar \
  --id SAMPLE_ID \
  --out-dir plots
```

Outputs from both workflows are local artifacts and ignored by git.
