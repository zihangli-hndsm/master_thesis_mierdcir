#!/usr/bin/env python3
"""Bootstrap confidence intervals for paired metric differences (RAW/FIXED/MULTI).

Pairs runs by seed: (s42, s123, [s2025]) per condition. For each dataset/metric,
computes bootstrap 95% CI over the mean of paired per-seed differences.
Seeds=2 or 3 -> n is tiny; results are illustrative, not formal inference.
"""
import json
import glob
import random
import statistics as st
import os

BASE = "/home/zihali/final_thesis"
SEEDS = ["s42", "s123", "s2025"]
CONDITIONS = ["raw", "fixed", "multi"]

# Reuse the combined metrics when present, else rebuild from per-condition JSONs.
ALL_METRICS_PATH = f"{BASE}/checkpoints/pair_control_all_metrics.json"
if os.path.exists(ALL_METRICS_PATH):
    combined = json.load(open(ALL_METRICS_PATH))
else:
    combined = {}

def load_condition_metrics():
    """Load metrics per (condition, seed) from eval JSONs / logs if combined missing."""
    if combined:
        return combined
    return {}

def metric_value(metrics, dataset_key, metric_key):
    return metrics.get(dataset_key, {}).get(metric_key)

def get_seed_metrics(cond, seed):
    ck = f"{cond}_{seed}"
    return combined.get(ck, {})

def bootstrap_ci(values, n_boot=10000, seed=0, alpha=0.05):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(st.mean(sample))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return lo, hi

def main():
    if not combined:
        print("No combined metrics file found; cannot run bootstrap.")
        return

    datasets = {
        "MTCIR test (source)": [("Recall@1", "R@1"), ("mAP", "mAP")],
        "MerdCIR": [("Recall@1", "R@1"), ("mAP", "mAP")],
        "FashionIQ": [("Recall@1", "R@1"), ("mAP", "mAP")],
        "CIRR recall (val)": [("Recall@1", "R@1"), ("Recall@5", "R@5")],
        "CIRR subset (val)": [("Recall_subset@1", "sub@1"), ("Recall_subset@3", "sub@3")],
    }
    keymap = {
        "MTCIR test (source)": "mtcir_test",
        "MerdCIR": "merdcir",
        "FashionIQ": "fashioniq",
        "CIRR recall (val)": "cirr_val_recall",
        "CIRR subset (val)": "cirr_val_subset",
    }

    # Collect available seeds per condition
    avail = {c: [s for s in SEEDS if get_seed_metrics(c, s)] for c in CONDITIONS}

    pairs = [("raw", "fixed"), ("raw", "multi"), ("multi", "fixed")]
    results = {}
    for a, b in pairs:
        common_seeds = [s for s in SEEDS if s in avail[a] and s in avail[b]]
        if len(common_seeds) < 2:
            print(f"\n== {a} vs {b}: only {len(common_seeds)} common seeds, skip ==")
            continue
        print(f"\n== {a} vs {b} (paired seeds: {common_seeds}) ==")
        for ds, metrics in datasets.items():
            block = keymap[ds]
            for key, label in metrics:
                diffs = []
                for s in common_seeds:
                    va = metric_value(get_seed_metrics(a, s), block, key)
                    vb = metric_value(get_seed_metrics(b, s), block, key)
                    if va is not None and vb is not None:
                        diffs.append(va - vb)
                if len(diffs) < 2:
                    continue
                mean_diff = st.mean(diffs) * 100
                ci = bootstrap_ci(diffs)
                if ci:
                    lo, hi = ci
                    print(f"  {ds} {label}: {mean_diff:+.2f} pp  "
                          f"[{lo*100:+.2f}, {hi*100:+.2f}]  (n={len(diffs)})")
                    results.setdefault(f"{a}_vs_{b}", {})[f"{ds}|{label}"] = {
                        "mean_diff_pp": mean_diff,
                        "ci95_lo_pp": lo * 100,
                        "ci95_hi_pp": hi * 100,
                        "n_seeds": len(diffs),
                    }

    out = f"{BASE}/checkpoints/pair_control_bootstrap_ci.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
