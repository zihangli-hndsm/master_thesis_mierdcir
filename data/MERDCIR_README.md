# MerdCIR Data Rewriting Pipeline

This directory now contains the MerdCIR framework implementation for rewriting the MTCIR dataset using Vision-Language Models deployed locally.

## Files
- `merdcir_prompts.py`: Defines the 6 discrete human-intent scenarios (Intent Randomness). 
- `merdcir_utils.py`: Extracts and drops nouns/prepositions to mimic natural concise human queries (Pragmatic Randomness).
- `rewrite_pipeline.py`: A high-throughput parallel VLM pipeline script utilizing `vLLM` to maximize the A100 GPU capability, interleaved with LMDB data fetching and saving outputs to JSONL.

## Installation Requirements
To run the pipeline on your A100, ensure you have a Linux environment (or WSL2) and install the following packages:
```bash
# Core inference engine (requires CUDA, Linux recommended)
pip install vllm transformers accelerate
# For Semantic POS dropping logic
pip install spacy
python -m spacy download en_core_web_sm
# Utilities
pip install Pillow lmdb tqdm
```

## Running the Pipeline
You can run the dataset rewriting by pointing `rewrite_pipeline.py` to your `MTCIR.jsonl` and LMDB data:

```bash
python data/rewrite_pipeline.py \
    --jsonl_path data/MTCIR/MTCIR.jsonl \
    --lmdb_path <PATH_TO_YOUR_LMDB_FILE> \
    --output_path data/MTCIR/MTCIR_merdcir_rewritten.jsonl \
    --model_name "Qwen/Qwen2-VL-7B-Instruct-AWQ" \
    --batch_size 64
```

### Notes & Optimization
- **Memory & VRAM**: The script uses AWQ quantization by default, so a 7B model takes ~6-8GB. An 80GB A100 can run extremely large batch sizes or high `tensor_parallel_size`. You can edit `tensor_parallel_size=1` in the script if you plan to launch this across multiple GPUs.
- **Batched Generation**: `vLLM` handles multi-modal inputs extremely efficiently.
- **Fail-safes**: If `spaCy` is missing, the code gracefully falls back to a simpler python string heuristic without crashing the long-running generation.
- Ensure the relative/absolute paths matches your exact workspace structure.
