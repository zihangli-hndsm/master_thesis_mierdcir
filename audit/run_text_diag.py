#!/usr/bin/env python3
"""MTCIR text-input diagnostics (plan §7.6):
  - baseline      : original modification text
  - shuffled-text : texts randomly permuted across queries
  - reference-only: modification text removed (empty string)
If R@1 stays abnormally high under shuffled/reference-only, the model may be
ignoring text / leaking via images or target mapping.
Runs on a subsample of pair_control/test.jsonl for speed.
"""
import argparse
import copy
import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from eval_checkpoint import EvalJsonDataset, evaluate

LMDB = "./data/MTCIR/images_224_lmdb"
SPLIT = "data/pair_control/test.jsonl"
K_LIST = (1, 5, 10, 50)


def make_loader(ds, batch_size=128):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                      pin_memory=True, collate_fn=EvalJsonDataset.collate_fn)


def permute_texts(items, rng):
    texts = [item.get("modification") for item in items]
    perm = list(range(len(texts)))
    rng.shuffle(perm)
    for i, item in enumerate(items):
        item["modification"] = texts[perm[i]]


def clear_texts(items):
    for item in items:
        item["modification"] = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", default="cross_attn_alpha")
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.full_model import ScheiCIR
    model = ScheiCIR(method=args.method, temperature=0.07).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=True)
    model.eval()

    base_ds = EvalJsonDataset(data_path=".", json_path=SPLIT, lmdb_path=LMDB, require_target=True)
    items = base_ds.data[: args.n_queries]
    base_ds.data = items
    print(f"n_queries: {len(items)}", flush=True)

    rng = random.Random(args.seed)
    results = {}

    results["baseline"] = evaluate(model, make_loader(base_ds), f"diag_base_{args.seed}", device, k_list=K_LIST)
    print("baseline:", results["baseline"], flush=True)

    shuf_ds = copy.deepcopy(base_ds)
    permute_texts(shuf_ds.data, rng)
    results["shuffled_text"] = evaluate(model, make_loader(shuf_ds), f"diag_shuf_{args.seed}", device, k_list=K_LIST)
    print("shuffled_text:", results["shuffled_text"], flush=True)

    ref_ds = copy.deepcopy(base_ds)
    clear_texts(ref_ds.data)
    results["reference_only"] = evaluate(model, make_loader(ref_ds), f"diag_ref_{args.seed}", device, k_list=K_LIST)
    print("reference_only:", results["reference_only"], flush=True)

    out = f"audit/text_diag_{args.seed}.json"
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "n_queries": len(items), "results": results}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
