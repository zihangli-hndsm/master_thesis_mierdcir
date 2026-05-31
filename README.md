# Final Thesis: Composed Image Retrieval Experiments

This repository contains code, notes, and evaluation utilities for thesis experiments on composed image retrieval (CIR), including MTCIR, MerdCIR, CIRR, FashionIQ, and LaSCo workflows.

The project includes:

- training and checkpoint evaluation scripts for the `ScheiCIR` model
- dataset loading and rewriting utilities under `data/`
- model components under `models/`
- human-evaluation tooling with a local Gradio app
- attention visualization and result-analysis scripts
- experiment notes in Markdown files

Large datasets, model checkpoints, generated JSON/JSONL files, Chroma databases, logs, and exported result artifacts are intentionally excluded from git. See `.gitignore` for the exact rules.

## License And Data Policy

Code in this repository is released under the MIT License. See `LICENSE`.

Third-party datasets are not redistributed in this repository. Generated data created by this project is intended to be released under CC BY 4.0 where the project author controls the relevant rights. See `DATASET_NOTICE.md` and `GENERATED_DATA_LICENSE.md`.

Training images used with this project come from the LLaVA-pretrain dataset available on Hugging Face. LLaVA-pretrain is a subset of Google's Conceptual Captions 3M (CC3M), and may contain harmful or sensitive content. Treat local datasets as uncurated web-scale data and apply appropriate safety review before use or sharing.

## Repository Layout

```text
.
├── app.py                         # Local Gradio app for human evaluation
├── train.py                       # Main training script
├── eval.py                        # Evaluation helper
├── eval_checkpoint.py             # Checkpoint evaluation on CIR datasets
├── evaluate_topk_checkpoints.py   # Batch evaluation for top-k checkpoints
├── visualize_attention.py         # Attention visualization script
├── analyze_human_rating_results.py
├── data/                          # Dataset code and preprocessing utilities
├── models/                        # Model and interaction modules
├── *.md                           # Notes, guidelines, and experiment reports
└── requirements.txt
```

Expected local-only artifact directories include:

```text
checkpoints/          # trained model checkpoints and metrics
chroma_db/            # persistent Chroma vector database files
logs/                 # training/evaluation logs
human_annotation_images/
human_rating_results/
plots/
reproduction_experiments/
```

## Setup

Create a virtual environment and install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository includes one broad `requirements.txt` for the main training, evaluation, visualization, and annotation workflows. For CUDA-specific PyTorch installation or optional VLM rewriting with `vllm`, see the notes in `REPRODUCE.md`.

For dataset restoration, LMDB creation, training, and evaluation commands, see `REPRODUCE.md`.

Tested environment:

```text
Python 3.13.12
CUDA 13.0
GPU: NVIDIA A100
```

## Common Workflows

Run training:

```bash
python train.py
```

Evaluate a checkpoint:

```bash
python eval_checkpoint.py --help
```

Evaluate top-k checkpoints:

```bash
python evaluate_topk_checkpoints.py --help
```

Start the human-evaluation app:

```bash
python app.py
```

Generate attention visualizations:

```bash
python visualize_attention.py --help
```

## Data And Checkpoints

The code expects datasets and LMDB image stores to be available locally under paths such as `data/CIRR`, `data/FashionIQ`, `data/MTCIR`, and `data/LaSCo`. These files are not committed because they are large generated or downloaded artifacts.

Likewise, model weights and checkpoint evaluation outputs should remain under `checkpoints/` or another ignored local directory.

## Notes

- Keep source code, scripts, and Markdown documentation in git.
- Keep dataset files, images, vector indexes, archives, logs, generated plots, CSV/PDF reports, and model checkpoints out of git.
- If a small JSON file later becomes source configuration rather than generated data, add an explicit exception for that file in `.gitignore`.
