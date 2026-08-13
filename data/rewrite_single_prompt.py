#!/usr/bin/env python3
"""Rewrite MTCIR with a SINGLE fixed prompt (no intent diversity) for controlled comparison.

Contrast with the full MerdCIR pipeline (rewrite_pipeline.py) which randomly selects
from 6 intent scenarios. This script uses one fixed generic prompt for every sample.

Usage:
    python rewrite_single_prompt.py \
        --input-jsonl part_0.jsonl \
        --output-jsonl part_0_fixed_prompt.jsonl \
        --model_name "Qwen/Qwen3.5-27B-FP8" \
        --batch_size 16

The output JSONL contains the original fields plus:
    - merdcir_modification: rewritten using the fixed prompt
    - merdcir_intent_scenario: always "fixed_generic"
"""

import os
import sys

conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
if conda_lib:
    os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import argparse
import json
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow running from project root or from data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.merdcir_utils import apply_pragmatic_randomness

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================================
# Single fixed prompt (no intent diversity)
# ============================================================================

FIXED_SYSTEM_PROMPT = (
    "You are an intelligent visual search assistant. "
    "The user wants to search for a new image starting from a reference image. "
    "You will be provided with two images: [Image 1] (the reference) and [Image 2] (the target). "
    "You must output a highly concise, natural, intent-oriented modification text "
    "that a human would write to transition from Image 1 to Image 2."
)

FIXED_USER_PROMPT = (
    "You are given two images. The first is the reference image, the second is the desired target image.\n"
    "The original basic description of the change was: '{original_modification}'.\n\n"
    "Your task is to re-write this into a natural human search query.\n"
    "Write a single concise sentence that captures the core visual change or intent.\n"
    "Focus on what the user WANTS (the target), not on listing every difference.\n"
    "Do NOT describe the reference image. Do NOT list what to remove/add/change one by one.\n\n"
    "Output ONLY the newly generated text query. Do not include quotes, explanations, or introductory phrases."
)

FIXED_SCENARIO = "fixed_generic"


def build_fixed_prompt(original_modification: str) -> list:
    """Build a prompt with the single fixed instruction."""
    user_text = FIXED_USER_PROMPT.format(original_modification=original_modification)
    messages = [
        {"role": "system", "content": FIXED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return messages


# ============================================================================
# Dataset (same logic as rewrite_pipeline.py, but reads from the part file)
# ============================================================================


def custom_collate_fn(batch):
    return batch


class RewriteDataset(Dataset):
    def __init__(self, lines: List[str], lmdb_path: str):
        self.lines = lines
        self.lmdb_path = lmdb_path
        self.env: Optional[lmdb.Environment] = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        item = json.loads(self.lines[idx])
        image_key1 = item["image"]
        image_key2 = item["target_image"]

        with self.env.begin(write=False) as txn:
            img_buf1 = txn.get(image_key1.encode("ascii"))
            img_buf2 = txn.get(image_key2.encode("ascii"))

        if img_buf1 is None or img_buf2 is None:
            return {"item": item, "img1": None, "img2": None, "orig_mod": ""}

        img1 = Image.open(BytesIO(img_buf1)).convert("RGB")
        img2 = Image.open(BytesIO(img_buf2)).convert("RGB")

        orig_mod = item.get("modifications", "")
        if isinstance(orig_mod, list):
            orig_mod = " ".join(str(s).strip(". ") for s in orig_mod if str(s).strip())

        return {"item": item, "img1": img1, "img2": img2, "orig_mod": orig_mod}


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite MTCIR with a single fixed prompt (no intent diversity)."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL (e.g. part_0.jsonl).")
    parser.add_argument("--lmdb-path", default="./data/MTCIR/images_224_lmdb", help="Path to MTCIR LMDB.")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-27B-FP8", help="Quantized Qwen model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Offline inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=0, help="Process only first N samples (0 = all).")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed ID in output file.")
    parser.add_argument(
        "--no-pragmatic-randomness",
        action="store_true",
        help="Skip the POS-dropping pragmatic randomness step.",
    )
    args = parser.parse_args()

    # --- vLLM setup ---
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
    except ImportError:
        logging.error("vLLM or transformers not found. Install with: pip install vllm transformers")
        return

    logging.info(f"Initializing VLM with model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        quantization="awq" if "AWQ" in args.model_name else None,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=250)

    # --- Resume logic ---
    processed_ids: set = set()
    open_mode = "w"
    if args.resume and os.path.exists(args.output_jsonl):
        open_mode = "a"
        logging.info(f"Scanning {args.output_jsonl} for processed IDs...")
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if "id" in item:
                        processed_ids.add(item["id"])
                except json.JSONDecodeError:
                    pass
        logging.info(f"Found {len(processed_ids)} already processed IDs — will skip.")

    # --- Load input ---
    lines = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("id") in processed_ids:
                    continue
                lines.append(line)
            except json.JSONDecodeError:
                pass

    if args.max_samples > 0:
        lines = lines[: args.max_samples]

    logging.info(f"Total entries to process: {len(lines)}")

    out_file = open(args.output_jsonl, open_mode, encoding="utf-8")

    dataset = RewriteDataset(lines, args.lmdb_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    # --- Batch rewrite loop ---
    for step, batch in enumerate(tqdm(dataloader, desc="Rewriting (fixed prompt)")):
        batch_inputs = []
        parsed_items = []

        for data in batch:
            item = data["item"]
            img1 = data["img1"]
            img2 = data["img2"]
            orig_mod = data["orig_mod"]

            parsed_items.append(item)

            if img1 is None or img2 is None:
                logging.warning(f"Skipping ID {item.get('id', '?')}: missing LMDB image.")
                batch_inputs.append(None)
                continue

            # Single fixed prompt (NO randomization of intent)
            messages = build_fixed_prompt(orig_mod)
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            batch_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": [img1, img2]},
            })

        valid_indices = [i for i, bp in enumerate(batch_inputs) if bp is not None]
        valid_inputs = [batch_inputs[i] for i in valid_indices]

        if not valid_inputs:
            continue

        outputs = llm.generate(valid_inputs, sampling_params, use_tqdm=False)

        # Periodic VRAM reporting
        if torch.cuda.is_available() and step % 500 == 0:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            logging.info(
                f"Batch {step} VRAM: {used_mem/1024**3:.2f}GB / {total_mem/1024**3:.2f}GB "
                f"({(used_mem/total_mem)*100:.1f}%)"
            )

        # Map results back and save
        for idx, generated_output in zip(valid_indices, outputs):
            raw_text = generated_output.outputs[0].text.strip()

            # Remove <think>...</think> blocks
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
            raw_text = re.sub(r"<think>.*", "", raw_text, flags=re.DOTALL).strip()

            # Apply pragmatic randomness (same as full MerdCIR pipeline)
            if not args.no_pragmatic_randomness:
                final_text = apply_pragmatic_randomness(raw_text, drop_prob=0.3)
            else:
                final_text = raw_text

            parsed_items[idx]["merdcir_modification"] = final_text
            parsed_items[idx]["merdcir_intent_scenario"] = FIXED_SCENARIO
            out_file.write(json.dumps(parsed_items[idx], ensure_ascii=False) + "\n")

        if (step + 1) % 1000 == 0:
            out_file.flush()
            os.fsync(out_file.fileno())
            logging.info(f"Checkpoint saved at step {step + 1}.")

    out_file.close()
    logging.info("Pipeline completed!")


if __name__ == "__main__":
    main()
