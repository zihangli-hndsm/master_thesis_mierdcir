#!/usr/bin/env python3
"""Freeze the CIRR-test submission checkpoint manifest (plan §2.1).

Selection rule (pre-registered, FOLLOWUP_EXPERIMENT_PLAN.md §5.1):
  c* = argmax common-dev mAP per condition from training_log_<cond>.json.
Records checkpoint path + SHA256 + training provenance for each of the
9 pair-control checkpoints (RAW/FIXED/MULTI x s42/s123/s2025).
"""
import glob
import hashlib
import json
import os

BASE = "/home/zihali/final_thesis"
CONDS = ["raw", "fixed", "multi"]
SEEDS = [42, 123, 2025]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def select_best(cond, seed):
    log_path = f"{BASE}/checkpoints/training_log_{cond}_s{seed}.json"
    if not os.path.isfile(log_path):
        return None, None, None
    entries = json.load(open(log_path))
    ranked = sorted(entries, key=lambda e: e.get("in_domain", {}).get("mAP", 0.0), reverse=True)
    for e in ranked:
        p = f"{BASE}/checkpoints/topk_pair_{cond}_s{seed}/topk_epoch_{e['epoch']:04d}_step_{e['step']:06d}_score_{e['selection_score']:.6f}.pth.tar"
        if os.path.isfile(p):
            return p, e.get("in_domain", {}).get("mAP"), e
    return None, None, None

manifest = {
    "protocol": "FOLLOWUP_EXPERIMENT_PLAN.md §5.1: c* = argmax common-dev mAP (in_domain)",
    "selection_data_used": "common-dev (MTCIR original text), pair_control/dev.jsonl",
    "test_data_used_for_selection": "NO (test split frozen 2026-08-08, never seen during selection)",
    "training_common": {
        "method": "merdcir_mlp_alpha (model_method=cross_attn_alpha)",
        "epochs": 3,
        "batch_size": 300,
        "backbone": "openai/clip-vit-base-patch32 (ViT-B/32)",
        "data": "pair_control strict intersection (train 255400 / dev 10000 / test 10000)",
        "text_field": "RAW=modification (MTCIR original), FIXED/MULTI=rewritten field per condition",
        "validation": "every 300 steps + epoch end",
    },
    "checkpoints": {},
}

for cond in CONDS:
    for seed in SEEDS:
        path, mAP, entry = select_best(cond, seed)
        if path is None:
            print(f"MISSING: {cond} s{seed}")
            continue
        rel = os.path.relpath(path, BASE)
        manifest["checkpoints"][f"{cond}_s{seed}"] = {
            "condition": cond,
            "seed": seed,
            "checkpoint": rel,
            "checkpoint_sha256": sha256(path),
            "common_dev_mAP": mAP,
            "selection_entry": {k: entry[k] for k in ("epoch", "step", "selection_score") if k in entry},
        }
        print(f"  {cond}_s{seed}: {rel} (common-dev mAP {mAP:.4f})")

out = f"{BASE}/cirr_test_results/checkpoint_manifest.json"
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote {out} ({len(manifest['checkpoints'])} checkpoints)")
