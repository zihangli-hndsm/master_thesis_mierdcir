#!/usr/bin/env python3
"""Compute descriptive CIRR per-automatic-intent hit rates from frozen exports."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_intent_coverage_v2 import classify  # noqa: E402

ROOT = HERE.parents[1]
PRED = ROOT / "narrative_followup/predictions/cirr_val"


def main() -> None:
    gt = json.loads((ROOT / "data/CIRR/cap.rc2.val.json").read_text(encoding="utf-8"))
    records = {str(row["pairid"]): row for row in gt}
    rows = []
    for condition in ("fixed", "multi"):
        for seed in ("42", "123", "2025"):
            for metric in ("recall", "recall_subset"):
                path = PRED / f"{condition}_s{seed}_{metric}.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                for pairid, row in records.items():
                    candidates = data.get(pairid, [])
                    if isinstance(candidates, dict):
                        candidates = candidates.get("candidates", [])
                    label, matched, confidence = classify(row.get("caption", ""))
                    target = row["target_hard"]
                    rows.append({
                        "condition": condition,
                        "seed": seed,
                        "metric": metric,
                        "pairid": pairid,
                        "intent": label,
                        "confidence": confidence,
                        "hit": int(target in candidates[: (1 if metric == "recall_subset" else 1)]),
                        "target": target,
                    })

    # The exported global list has 50 candidates; its first item is global R@1.
    # The exported subset list has three candidates; its first item is subset R@1.
    out = ROOT / "narrative_followup/metrics/intent_performance_cirr_v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["metric"], row["intent"])].append(row["hit"])
    summary = []
    for (condition, metric, intent), hits in sorted(grouped.items()):
        summary.append({
            "condition": condition,
            "metric": metric,
            "intent": intent,
            "n": len(hits),
            "hit_at_1": sum(hits) / len(hits) if hits else 0.0,
            "automatic_only": True,
        })
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    report = ROOT / "narrative_followup/reports/intent_performance_cirr_v2.md"
    report_lines = [
        "# CIRR per-intent performance v2",
        "",
        "These hit rates use automatic lexical labels and frozen prediction exports. "
        "They are descriptive only; manual label validation is still required before "
        "any intent-specific mechanism claim.",
        "",
        "`n` counts query-seed observations (4,181 queries × 3 seeds). The metric is "
        "the first-ranked target hit rate for the corresponding global or candidate-subset export.",
        "",
        "| Condition | Metric | Automatic intent | n | Hit@1 |",
        "|---|---|---|---:|---:|",
    ]
    report_lines.extend(
        f"| {row['condition']} | {row['metric']} | {row['intent']} | {row['n']} | {row['hit_at_1']:.4f} |"
        for row in summary
    )
    report_lines.extend([
        "",
        "No causal or category-specific claim should be made until the manual review sheet is completed.",
        "",
    ])
    report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"wrote {out} rows={len(summary)}")


if __name__ == "__main__":
    main()
