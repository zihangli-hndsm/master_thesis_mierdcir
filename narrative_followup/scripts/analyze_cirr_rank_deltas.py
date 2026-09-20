#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/zihali/final_thesis")
OUT = ROOT / "narrative_followup"
PRED = OUT / "predictions/cirr_val"
GT = ROOT / "data/CIRR/cap.rc2.val.json"


def rank(pred: list[str], target: str, miss_rank: int) -> int:
    try:
        return pred.index(target) + 1
    except ValueError:
        return miss_rank


def main() -> None:
    gt = json.loads(GT.read_text())
    predictions = {}
    for cond in ("fixed", "multi"):
        for seed in (42, 123, 2025):
            for metric in ("recall", "recall_subset"):
                path = PRED / f"{cond}_s{seed}_{metric}.json"
                predictions[(cond, seed, metric)] = json.loads(path.read_text())
    rows = []
    for item in gt:
        pairid = str(item["pairid"])
        target = item["target_hard"]
        reference = item.get("reference")
        for seed in (42, 123, 2025):
            base = {}
            for cond in ("fixed", "multi"):
                for metric, miss in (("recall", 51), ("recall_subset", 4)):
                    pred = predictions[(cond, seed, metric)][pairid]
                    base[(cond, metric)] = rank(pred, target, miss)
            rows.append(
                {
                    "pairid": pairid,
                    "reference": reference,
                    "target": target,
                    "seed": seed,
                    "fixed_global_rank_censored_51": base[("fixed", "recall")],
                    "multi_global_rank_censored_51": base[("multi", "recall")],
                    "delta_global_fixed_minus_multi": base[("fixed", "recall")] - base[("multi", "recall")],
                    "fixed_subset_rank_censored_4": base[("fixed", "recall_subset")],
                    "multi_subset_rank_censored_4": base[("multi", "recall_subset")],
                    "delta_subset_fixed_minus_multi": base[("fixed", "recall_subset")] - base[("multi", "recall_subset")],
                }
            )

    fields = list(rows[0])
    per_seed = OUT / "metrics/cirr_val_rank_deltas_per_seed.csv"
    per_seed.parent.mkdir(parents=True, exist_ok=True)
    with per_seed.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # Aggregate by query: mean rank among the three fixed training seeds.
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pairid"], []).append(row)
    agg = []
    for pairid, items in grouped.items():
        x = items[0].copy()
        for key in fields:
            if key in {"seed", "pairid"}:
                continue
            if key.startswith("fixed_") or key.startswith("multi_") or key.startswith("delta_"):
                x[key] = sum(float(i[key]) for i in items) / len(items)
        x["seed"] = "mean_3_seeds"
        agg.append(x)
    agg_path = OUT / "metrics/cirr_val_rank_deltas_aggregated.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(agg)

    summary = {}
    for metric, key in (("global", "delta_global_fixed_minus_multi"), ("subset", "delta_subset_fixed_minus_multi")):
        vals = [float(x[key]) for x in rows]
        summary[metric] = {
            "n_query_seed": len(vals),
            "multi_better_count": sum(v > 0 for v in vals),
            "fixed_better_count": sum(v < 0 for v in vals),
            "ties": sum(v == 0 for v in vals),
            "mean_fixed_minus_multi_censored_rank": sum(vals) / len(vals),
        }
    (OUT / "metrics/cirr_val_rank_delta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
