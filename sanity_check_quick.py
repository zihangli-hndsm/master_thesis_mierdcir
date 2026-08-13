#!/usr/bin/env python3
"""Fast data quality scan for MTCIR - skips full Dataset loading."""
import json, lmdb, sys
from collections import Counter
from io import BytesIO
from PIL import Image

LMDB_PATH = "/home/zihali/final_thesis/data/MTCIR/images_224_lmdb"
JSONL = "/home/zihali/final_thesis/data/mtcir_np/merged.jsonl"
N_SCAN = 10000

print(f"Scanning {N_SCAN} records from {JSONL}...")

env = lmdb.open(LMDB_PATH, readonly=True, lock=False)

stats = Counter()
bad_images = 0
bad_nps = 0
dup_targets = Counter()
text_lens = []
np_lens = []

with open(JSONL) as f:
    for i, line in enumerate(f):
        if i >= N_SCAN:
            break
        rec = json.loads(line)

        ref = rec.get("image", "")
        tgt = rec.get("target_img", "")
        mod = rec.get("modification", "")
        nps = rec.get("nps", [])

        stats["total"] += 1
        text_lens.append(len(str(mod)))
        np_lens.append(len(nps))

        if ref == tgt:
            stats["ref_equals_tgt"] += 1
        dup_targets[tgt] += 1

        if not str(mod).strip():
            stats["empty_mod"] += 1
        if len(nps) == 0:
            stats["no_nps"] += 1

        # Check NP spans
        for np_item in nps:
            if len(np_item) >= 2 and isinstance(np_item[1], (list, tuple)) and len(np_item[1]) == 2:
                if np_item[1][1] > 77:
                    bad_nps += 1

        # Image spot check (every 500th)
        if i % 500 == 0:
            with env.begin(write=False, buffers=True) as txn:
                for key in [ref, tgt]:
                    buf = txn.get(key.encode("ascii"))
                    if buf is None:
                        bad_images += 1
                    else:
                        try:
                            img = Image.open(BytesIO(buf))
                            w, h = img.size
                            if w < 50 or h < 50:
                                bad_images += 1
                        except Exception:
                            bad_images += 1

env.close()

print(f"\n=== Results (n={stats['total']}) ===")
print(f"  ref == target:        {stats['ref_equals_tgt']}")
print(f"  empty modifications:  {stats['empty_mod']}")
print(f"  no NPs:               {stats['no_nps']}")
print(f"  bad NP spans (>77):   {bad_nps}")
print(f"  bad images:           {bad_images}")
print(f"  text length:  min={min(text_lens):.0f}  mean={sum(text_lens)/len(text_lens):.0f}  max={max(text_lens):.0f}")
print(f"  NP count:     min={min(np_lens):.0f}  mean={sum(np_lens)/len(np_lens):.1f}  max={max(np_lens):.0f}")

# Duplicate target analysis
repeated = sum(1 for v in dup_targets.values() if v > 1)
print(f"  distinct targets:     {len(dup_targets)}/{stats['total']} ({len(dup_targets)/stats['total']:.1%})")
print(f"  targets with >1 ref:  {repeated}")

top_dups = dup_targets.most_common(5)
print(f"  most repeated targets: {top_dups}")

# Quick MerdCIR check
M_JSONL = "/home/zihali/final_thesis/data/merdcir_np/test_train.jsonl"
print(f"\n=== MerdCIR quick check ===")
with open(M_JSONL) as f:
    m_stats = Counter()
    m_text_lens = []
    for i, line in enumerate(f):
        if i >= 1000:
            break
        rec = json.loads(line)
        mod = rec.get("modification", "")
        m_text_lens.append(len(str(mod)))
print(f"  text length:  min={min(m_text_lens):.0f}  mean={sum(m_text_lens)/len(m_text_lens):.0f}  max={max(m_text_lens):.0f}")
print(f"  total records: expecting 280136")

# Token length warning from CLIP
print(f"\n=== Token length warning ===")
from transformers import CLIPTokenizerFast
tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
overflow = 0
with open(JSONL) as f:
    for i, line in enumerate(f):
        if i >= 5000:
            break
        rec = json.loads(line)
        mod = rec.get("modification", "")
        ids = tok(mod, truncation=True, max_length=77, return_tensors=None)["input_ids"]
        if len(ids) >= 77:
            overflow += 1
print(f"  texts truncated at 77 tokens: {overflow}/5000 ({overflow/50:.1f}%)")
print(f"  (CLIP warning '78 > 77' is expected for long texts)")

print("\nDone.")
