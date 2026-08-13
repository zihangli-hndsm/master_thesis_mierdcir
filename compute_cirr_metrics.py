#!/usr/bin/env python3
"""Compute CIRR recall / recall_subset metrics from prediction files + ground truth.

Fixed 2026-08-13: recall_subset now uses target_hard when available (matching
evaluate_cirr_subset); the img_set-members fallback is kept only for label-free
test1 predictions. Added --gt to support val rehearsal against cap.rc2.val.json.
"""
import json
import glob
import os
import argparse

BASE = "/home/zihali/final_thesis"

def compute(pred_path, gt, mode):
    """mode: 'recall' (global gallery) or 'recall_subset' (within img_set)."""
    pred = json.load(open(pred_path))
    ranks = []
    for qid_str, ranking in pred.items():
        if qid_str in ("version", "metric"):
            continue
        qid = int(qid_str)
        rec = gt.get(qid)
        if rec is None:
            continue
        target = rec.get("target_hard")
        if target is not None:
            rank = None
            for i, img in enumerate(ranking):
                if img == target:
                    rank = i + 1
                    break
            ranks.append(rank)
            continue
        # label-free (test1): fall back to img_set members
        members = set(rec.get("img_set", {}).get("members", []))
        rank = None
        for i, img in enumerate(ranking):
            if img in members:
                rank = i + 1
                break
        ranks.append(rank)

    n = len(ranks)
    valid = [r for r in ranks if r is not None]
    out = {
        "n_queries": n,
        "n_ranked": len(valid),
        "R@1": sum(1 for r in valid if r <= 1) / n if n else 0,
        "R@5": sum(1 for r in valid if r <= 5) / n if n else 0,
        "R@10": sum(1 for r in valid if r <= 10) / n if n else 0,
        "R@50": sum(1 for r in valid if r <= 50) / n if n else 0,
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=f"{BASE}/data/CIRR/cap.rc2.test1.json",
                    help="ground-truth captions JSON (default test1; pass cap.rc2.val.json for rehearsal)")
    ap.add_argument("--pred-dir", default=None,
                    help="directory of prediction JSONs; defaults to scanning checkpoints/topk_pair_*")
    args = ap.parse_args()
    gt = json.load(open(args.gt))
    gt_by_id = {r["pairid"]: r for r in gt}

    out = {}
    if args.pred_dir:
        preds = sorted(glob.glob(f"{args.pred_dir}/*_recall*.json"))
    else:
        preds = []
        for ckpt_dir in sorted(glob.glob(f"{BASE}/checkpoints/topk_pair_*")):
            cond = os.path.basename(ckpt_dir)
            entry = {}
            for mode in ("recall", "recall_subset"):
                p = f"{ckpt_dir}/{cond}_cirr_{mode}.json"
                if os.path.exists(p):
                    entry[mode] = compute(p, gt_by_id, mode)
            out[cond] = entry
            print(f"=== {cond} ===")
            print(json.dumps(entry, indent=2))
        with open(f"{BASE}/checkpoints/cirr_metrics.json", "w") as f:
            json.dump(out, f, indent=2)
        return

    # explicit pred files
    for p in preds:
        mode = "recall" if "recall_subset" not in p else "recall_subset"
        out[p] = compute(p, gt_by_id, mode)
        print(f"=== {p} ===")
        print(json.dumps(out[p], indent=2))
    with open(f"{BASE}/audit/cirr_pred_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
