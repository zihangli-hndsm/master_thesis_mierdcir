#!/usr/bin/env python3
"""Rewrite MTCIR with 6 SURFACE templates expressing the SAME generic retrieval intent.

Contrast with:
  - rewrite_single_prompt.py : ONE fixed generic prompt  (-> FIXED condition)
  - rewrite_pipeline.py      : 6 SEMANTIC intent scenarios (-> MULTI condition)
This script varies only the linguistic SURFACE (sentence frame), not the intent.

Comparison logic (FOLLOWUP_EXPERIMENT_PLAN.md §4.2):
  - FIXED vs SURFACE-DIVERSE  : effect of pure wording diversity
  - SURFACE-DIVERSE vs MULTI  : incremental effect of semantic intent diversity

Usage:
    python data/rewrite_surface_diverse.py \
        --input-jsonl pair_control/train.jsonl \
        --output-jsonl pair_control/train_surfacediverse_raw.jsonl \
        --model_name "Qwen/Qwen3.5-27B-FP8" \
        --batch-size 64 \
        --max-samples 50000
"""
import argparse
import json
import logging
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import lmdb
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.merdcir_utils import apply_pragmatic_randomness

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================================
# 6 surface templates, all expressing the same generic retrieval intent
# ("find a similar image with the described modification")
# ============================================================================

SURFACE_FRAMES = [
    "Show me a photo like this one, where {change}",
    "I am looking for a picture similar to this, but {change}",
    "Find an image that matches this one after {change}",
    "I want to see this scene with {change}",
    "Retrieve a photo that shows this image with {change}",
    "A version of this image where {change}",
]

SYSTEM_PROMPT = (
    "You are an intelligent visual search assistant. "
    "The user wants to search for a new image starting from a reference image. "
    "You will be provided with two images: [Image 1] (the reference) and [Image 2] (the target). "
    "You must output a highly concise, natural, intent-oriented modification text "
    "that a human would write to transition from Image 1 to Image 2."
)


def build_prompt(original_modification: str, frame: str) -> list:
    user_text = (
        "You are given two images. The first is the reference image, the second is the desired target image.\n"
        f"The original basic description of the change was: '{original_modification}'.\n\n"
        "Your task is to re-write this into a natural human search query.\n"
        "Write EXACTLY ONE sentence using this sentence frame:\n\n"
        f"    {frame}\n\n"
        "Fill in the '{change}' placeholder with the core visual modification "
        "(what the user wants in the target image).\n"
        "Do NOT describe the reference image. Do NOT list every difference.\n"
        "Output ONLY the completed sentence, keeping the frame wording intact."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def pick_frame(item_id: str) -> int:
    """Deterministic frame assignment: hash(id) % 6 -> balanced, reproducible."""
    return sum(ord(c) for c in str(item_id)) % len(SURFACE_FRAMES)


def custom_collate_fn(batch):
    return batch


class RewriteDataset(Dataset):
    """Reads pair_control/train.jsonl (fields: image / target_img / modification)."""

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
        image_key2 = item.get("target_img") or item.get("target_image")

        with self.env.begin(write=False) as txn:
            img_buf1 = txn.get(image_key1.encode("ascii"))
            img_buf2 = txn.get(image_key2.encode("ascii")) if image_key2 else None

        if img_buf1 is None or img_buf2 is None:
            return {"item": item, "img1": None, "img2": None, "orig_mod": ""}

        img1 = Image.open(BytesIO(img_buf1)).convert("RGB")
        img2 = Image.open(BytesIO(img_buf2)).convert("RGB")

        orig_mod = item.get("modification", "")
        if isinstance(orig_mod, list):
            orig_mod = " ".join(str(s).strip(". ") for s in orig_mod if str(s).strip())

        return {"item": item, "img1": img1, "img2": img2, "orig_mod": orig_mod}


def main():
    parser = argparse.ArgumentParser(description="Rewrite MTCIR with 6 surface templates (same generic intent).")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--lmdb-path", default="./data/MTCIR/images_224_lmdb")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-27B-FP8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Process only first N samples (0 = all).")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed ID in output file.")
    parser.add_argument("--no-pragmatic-randomness", action="store_true")
    args = parser.parse_args()

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

    for step, batch in enumerate(tqdm(dataloader, desc="Rewriting (surface-diverse)")):
        batch_inputs = []
        parsed_items = []

        for data in batch:
            item = data["item"]
            img1 = data["img1"]
            img2 = data["img2"]
            orig_mod = data["orig_mod"]

            parsed_items.append(item)

            if img1 is None or img2 is None or not orig_mod:
                logging.warning(f"Skipping ID {item.get('id', '?')}: missing LMDB image or text.")
                batch_inputs.append(None)
                continue

            frame_idx = pick_frame(item.get("id", ""))
            frame = SURFACE_FRAMES[frame_idx]
            messages = build_prompt(orig_mod, frame)
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": [img1, img2]},
                "frame_idx": frame_idx,
            })

        valid_indices = [i for i, bp in enumerate(batch_inputs) if bp is not None]
        valid_inputs = [batch_inputs[i] for i in valid_indices]

        if not valid_inputs:
            continue

        outputs = llm.generate(valid_inputs, sampling_params, use_tqdm=False)

        if torch.cuda.is_available() and step % 500 == 0:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            logging.info(
                f"Batch {step} VRAM: {used_mem/1024**3:.2f}GB / {total_mem/1024**3:.2f}GB "
                f"({(used_mem/total_mem)*100:.1f}%)"
            )

        for idx, generated_output in zip(valid_indices, outputs):
            raw_text = generated_output.outputs[0].text.strip()
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
            raw_text = re.sub(r"<think>.*", "", raw_text, flags=re.DOTALL).strip()

            if not args.no_pragmatic_randomness:
                final_text = apply_pragmatic_randomness(raw_text, drop_prob=0.3)
            else:
                final_text = raw_text

            parsed_items[idx]["merdcir_modification"] = final_text
            parsed_items[idx]["merdcir_intent_scenario"] = (
                f"surface_diverse_{batch_inputs[idx]['frame_idx'] + 1}"
            )
            out_file.write(json.dumps(parsed_items[idx], ensure_ascii=False) + "\n")

        if (step + 1) % 1000 == 0:
            out_file.flush()
            os.fsync(out_file.fileno())
            logging.info(f"Checkpoint saved at step {step + 1}.")

    out_file.close()
    logging.info("Pipeline completed!")


if __name__ == "__main__":
    main()
