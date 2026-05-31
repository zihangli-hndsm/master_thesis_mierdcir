import os
import sys

# 假设你的 conda 环境路径
conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
# 强制插入系统路径的第一位
os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
import json
import logging
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import lmdb
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

from merdcir_prompts import get_random_intent_prompt, build_qwen2vl_prompt
from merdcir_utils import apply_pragmatic_randomness

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_collate_fn(batch):
    return batch

class RewriteDataset(Dataset):
    def __init__(self, lines, lmdb_path):
        self.lines = lines
        self.lmdb_path = lmdb_path
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
            
        item = json.loads(self.lines[idx])
        
        image_key1 = item['image']
        image_key2 = item['target_img']
        
        with self.env.begin(write=False) as txn:
            img_buf1 = txn.get(image_key1.encode('ascii'))
            img_buf2 = txn.get(image_key2.encode('ascii'))
            
        if img_buf1 is None or img_buf2 is None:
            return {'item': item, 'img1': None, 'img2': None, 'orig_mod': item.get('modification', '')}
            
        img1 = Image.open(BytesIO(img_buf1)).convert('RGB')
        img2 = Image.open(BytesIO(img_buf2)).convert('RGB')
        
        orig_mod = item.get('modification', '')
        if isinstance(orig_mod, list):
            orig_mod = " ".join(orig_mod)
            
        return {'item': item, 'img1': img1, 'img2': img2, 'orig_mod': orig_mod}

def main():
    parser = argparse.ArgumentParser(description="Rewrite MTCIR dataset using MerdCIR intent logic and VLM.")
    parser.add_argument("--jsonl_path", type=str, default="./MTCIR/mtcir.jsonl", help="Path to original MTCIR.jsonl")
    parser.add_argument("--lmdb_path", type=str, default="./MTCIR/images_224_lmdb", help="Path to MTCIR LMDB")
    parser.add_argument("--output_path", type=str, default="rewrite_sample10p.jsonl", help="Path to output rewrited jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-27B-FP8", help="Quantized Qwen2-VL model path")
    parser.add_argument("--batch_size", type=int, default=16, help="Offline inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    args = parser.parse_args()

    # Import vLLM (ensure it is executed on standard A100 environment)
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
    except ImportError:
        logging.error("vLLM or transformers not found. Please `pip install vllm transformers` to use the A100 inference framework.")
        return

    # Initialize VLM Engine (Parallel execution on A100 via vLLM)
    logging.info(f"Initializing VLM with model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        quantization="awq" if "AWQ" in args.model_name else None,
        tensor_parallel_size=1,     # Can increase to 2 or 4 if using multiple GPUs
        max_model_len=4096,         # Context length limit
        gpu_memory_utilization=0.90 # Best for A100 high throughput
    )
    
    # We use AutoProcessor to apply chat templates correctly for Qwen2-VL
    processor = AutoProcessor.from_pretrained(args.model_name)
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=250  # Instructions should be concise
    )

    # Resume Logic: check what has already been processed
    processed_ids = set()
    open_mode = 'w'
    if os.path.exists(args.output_path):
        open_mode = 'a'
        logging.info(f"Found existing output at {args.output_path}. Scanning for processed IDs to resume...")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    if 'id' in item:
                        processed_ids.add(item['id'])
                except json.JSONDecodeError:
                    pass
        logging.info(f"Found {len(processed_ids)} already processed items. They will be skipped.")

    # Load remaining input
    lines = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                if item.get('id') not in processed_ids:
                    lines.append(line)
            except json.JSONDecodeError:
                pass
    
    logging.info(f"Total entries remaining to process: {len(lines)}")

    out_file = open(args.output_path, open_mode, encoding='utf-8')

    dataset = RewriteDataset(lines, args.lmdb_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn, 
        shuffle=False
    )

    # Batch processing loop
    # We yield batches of inputs to vLLM logic
    for step, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        batch_inputs = []
        parsed_items = []
        
        for data in batch:
            item = data['item']
            img1 = data['img1']
            img2 = data['img2']
            orig_mod = data['orig_mod']
            
            parsed_items.append(item)
            
            if img1 is None or img2 is None:
                logging.warning(f"Skipping ID {item.get('id', 'Unknown')} due to missing image in LMDB.")
                batch_inputs.append(None)
                continue
                
            # Rule Randomization (Intent)
            intent_key, intent_prompt = get_random_intent_prompt()
            
            # Record chosen intent for trackability
            item['merdcir_intent_scenario'] = intent_key
            
            # Construct Prompt Messages
            messages = build_qwen2vl_prompt(orig_mod, intent_prompt)
            prompt_text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # VLLM requires proper multi-modal layout for API
            batch_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": [img1, img2]}
            })

        # Filter out bad valid bounds
        valid_indices = [idx for idx, bp in enumerate(batch_inputs) if bp is not None]
        valid_inputs = [batch_inputs[idx] for idx in valid_indices]
        
        if not valid_inputs:
            continue
            
        # Parallel generation heavily sped up by A100 VLLM
        outputs = llm.generate(valid_inputs, sampling_params, use_tqdm=False)

        # Monitor VRAM usage to help adjust batch_size for A100
        if torch.cuda.is_available() and step % 500 == 0:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            logging.info(f"Batch {step} VRAM Usage: {used_mem / 1024**3:.2f}GB / {total_mem / 1024**3:.2f}GB ({(used_mem/total_mem)*100:.1f}%)")

        import re
        # Map back to items and save
        for idx, generated_output in zip(valid_indices, outputs):
            raw_text = generated_output.outputs[0].text.strip()
            
            # Remove <think>...</think> blocks if any exist
            raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
            # Remove unclosed <think> blocks (if max_tokens cuts it off)
            raw_text = re.sub(r'<think>.*', '', raw_text, flags=re.DOTALL).strip()
            
            # 2. Rule Randomization (Pragmatic Randomness)
            pragmatic_text = apply_pragmatic_randomness(raw_text, drop_prob=0.3)
            
            # Save newly rewritten instructions
            parsed_items[idx]['merdcir_modification'] = pragmatic_text
            out_file.write(json.dumps(parsed_items[idx], ensure_ascii=False) + '\n')
            
        # Periodic flush every 10,000 steps to save disk I/O overhead
        if (step + 1) % 1000 == 0:
            out_file.flush()
            os.fsync(out_file.fileno())  # Ensure it is written to disk hardware
            logging.info(f"Checkpoint saved at step {step + 1}. Data synced to disk.")
            
    out_file.close()
    logging.info("Pipeline processing completed successfully!")

if __name__ == "__main__":
    main()
