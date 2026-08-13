#!/usr/bin/env python3
"""Parse eval_followup log + CIRR JSONs into a per-condition summary table.

Reads:
  - logs/eval_followup_*.log  (MTCIR/MerdCIR/FashionIQ metrics, printed to stdout)
  - checkpoints/topk_pair_*/*_cirr_recall*.json (CIRR prediction files)
Writes:
  - checkpoints/pair_control_eval_summary.json
"""
import json
import re
import glob
import os

BASE = "/home/zihali/final_thesis"
LOG = sorted(glob.glob(f"{BASE}/logs/eval_followup_*.log"))[-1]

CONDITIONS = ["raw_s42", "raw_s123", "fixed_s42", "fixed_s123", "multi_s42", "multi_s123"]

def parse_log(path):
    """Return {condition: {dataset: {metric: value}}}"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    content = content.replace("\r", "\n")
    lines = content.split("\n")

    results = {}
    current_ckpt = None
    current_dataset = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.search(r"Evaluating: (\S+) on (\w+)", line)
        if m:
            current_ckpt = m.group(1)
            current_dataset = m.group(2)
            i += 1
            continue
        if line.startswith("Dataset:"):
            # Dataset: MTCIR / MerdCIR / FashionIQ/dress etc
            ds_name = line.split(":", 1)[1].strip()
            # Only capture full blocks for MTCIR/MerdCIR and FashionIQ average
            if ds_name in ("MTCIR", "MerdCIR", "FashionIQ/average"):
                current_dataset = ds_name
            i += 1
            continue
        if current_ckpt and current_dataset and line.startswith(("Recall@", "mAP")):
            cond = None
            for c in CONDITIONS:
                if c in current_ckpt:
                    cond = c
                    break
            if cond is None:
                i += 1
                continue
            key, _, val = line.partition(":")
            results.setdefault(cond, {}).setdefault(current_dataset, {})[key.strip()] = float(val)
        i += 1
    return results

def parse_cirr(cond):
    """Read CIRR recall + recall_subset JSONs; extract top-1 scores if present."""
    out = {}
    for metric in ["recall", "recall_subset"]:
        p = f"{BASE}/checkpoints/topk_pair_{cond}/topk_pair_{cond}_cirr_{metric}.json"
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            out[metric] = summarize_cirr(data)
    return out

def summarize_cirr(data):
    """data is a dict of query->rankings or similar; try common structures."""
    if isinstance(data, dict):
        # Look for summary keys
        summary = {}
        for k in ("Recall@1", "Recall@5", "Recall@10", "Recall@50", "R@1", "mAP", "recall@1"):
            if k in data:
                summary[k] = data[k]
        if summary:
            return summary
        # Per-query dict: {qid: rank or {rank: r}} -> compute R@1
        ranks = []
        for qid, v in data.items():
            if isinstance(v, dict) and "rank" in v:
                ranks.append(v["rank"])
            elif isinstance(v, (int, float)):
                ranks.append(v)
        if ranks:
            n = len(ranks)
            return {
                "n_queries": n,
                "R@1": sum(1 for r in ranks if r <= 1) / n,
                "R@5": sum(1 for r in ranks if r <= 5) / n,
                "R@10": sum(1 for r in ranks if r <= 10) / n,
                "R@50": sum(1 for r in ranks if r <= 50) / n,
            }
    return {"note": "unparsed structure"}

def main():
    log_results = parse_log(LOG)
    summary = {}
    for cond in CONDITIONS:
        entry = {"log_metrics": log_results.get(cond, {})}
        entry["cirr"] = parse_cirr(cond)
        summary[cond] = entry

    out_path = f"{BASE}/checkpoints/pair_control_eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")
    for cond in CONDITIONS:
        print(f"\n=== {cond} ===")
        print(json.dumps(summary[cond], indent=2)[:600])

if __name__ == "__main__":
    main()
