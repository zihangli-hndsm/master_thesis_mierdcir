#!/usr/bin/env python3
"""Merge s2025 eval results into pair_control_all_metrics.json and rerun bootstrap.

Parses logs/eval_s2025_*.log for ALL metrics (MTCIR/MerdCIR/FashionIQ + CIRR val recall/subset),
mirroring the formats proven in eval_followup_*.log and eval_cirr_subset2_*.log.
"""
import json
import glob
import os
import re

BASE = "/home/zihali/final_thesis"

def parse_eval_log(path):
    with open(path, encoding="utf-8") as f:
        content = f.read().replace("\r", "\n")
    lines = content.split("\n")
    results = {}
    cur_cond = None
    cur_ds = None
    for line in lines:
        line = line.strip()
        m = re.search(r"===\s*(\w+)\s*===", line)
        if m:
            cur_cond = m.group(1)
            cur_ds = None
            continue
        if cur_cond and line.startswith("Dataset:"):
            ds_name = line.split(":", 1)[1].strip()
            # CIRR (global recall) and CIRR/subset are both emitted by eval_checkpoint
            cur_ds = ds_name
            continue
        if cur_cond and cur_ds and line.startswith(("Recall@", "Recall_subset@", "mAP")):
            key, _, val = line.partition(":")
            results.setdefault(cur_cond, {}).setdefault(cur_ds, {})[key.strip()] = float(val)
    return results

def main():
    combined_path = f"{BASE}/checkpoints/pair_control_all_metrics.json"
    combined = json.load(open(combined_path)) if os.path.exists(combined_path) else {}

    logs = sorted(glob.glob(f"{BASE}/logs/eval_s2025_*.log"))
    if not logs:
        print("No eval_s2025 logs found")
        return
    # Parse ALL eval_s2025 logs and merge their conditions (each run covers a subset).
    log_metrics = {}
    for lp in logs:
        part = parse_eval_log(lp)
        for cond, blocks in part.items():
            log_metrics.setdefault(cond, {}).update(blocks)
    print("Parsed conditions:", list(log_metrics.keys()))

    # CIRR global-recall eval prints "Dataset: CIRR/recall" (val set has labels),
    # recall_subset prints "Dataset: CIRR/subset".
    mapping = {
        "MTCIR": "mtcir_test",
        "MerdCIR": "merdcir",
        "FashionIQ/average": "fashioniq",
        "CIRR/recall": "cirr_val_recall",
        "CIRR/subset": "cirr_val_subset",
    }

    merged = []
    for cond in ["multi_s2025", "fixed_s2025", "raw_s2025"]:
        if cond not in log_metrics:
            print(f"SKIP (no eval output): {cond}")
            continue
        if cond not in combined:
            combined[cond] = {}
        lm = log_metrics.get(cond, {})
        for ds_name, key in mapping.items():
            if ds_name in lm and lm[ds_name]:
                combined[cond][key] = lm[ds_name]
        merged.append(cond)

    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Updated {combined_path}")
    for cond in merged:
        print(f"\n{cond}:")
        print(json.dumps(combined.get(cond, {}), indent=2)[:800])

if __name__ == "__main__":
    main()
