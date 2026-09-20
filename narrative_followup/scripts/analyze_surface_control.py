#!/usr/bin/env python3
"""Aggregate the matched Surface-50K MTCIR/CIRR evaluations."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "narrative_followup/metrics"
GT = json.loads((ROOT / "data/CIRR/cap.rc2.val.json").read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    mtcir = []
    for path in sorted(METRICS.glob("surface50k_query_*.json")):
        stem = path.stem.removeprefix("surface50k_query_")
        condition, seed, style = stem.rsplit("_", 2)
        data = json.loads(path.read_text())
        row = {"condition": condition, "seed": int(seed[1:]), "query_style": style, "source": str(path)}
        row.update(data["metrics"])
        mtcir.append(row)
    mtcir_path = METRICS / "surface_control_mtcir_matrix.csv"
    fields = ["condition", "seed", "query_style", "Recall@1", "Recall@5", "Recall@10", "Recall@50", "mAP", "source"]
    with mtcir_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(mtcir)

    grouped = defaultdict(list)
    for row in mtcir:
        grouped[(row["condition"], row["query_style"])].append(row)
    lines = [
        "# Surface-50K matched-control results",
        "",
        "The formal control trains Fixed, deterministic Surface-v2 and Multi on the same frozen 50K pair IDs with the same optimizer schedule. Evaluation uses the held-out 10K test pairs and varies only query style.",
        "",
        f"MTCIR matrix cells available: **{len(mtcir)}/36**.",
        "",
    ]
    if grouped:
        for metric in ("Recall@1", "Recall@5", "Recall@10", "Recall@50", "mAP"):
            lines += [f"## {metric}", "", "| train condition \\ query style | Raw | Fixed | Multi | Surface |", "|---|---:|---:|---:|---:|"]
            for condition in ("fixed", "surface", "multi"):
                cells = []
                for style in ("raw", "fixed", "multi", "surface"):
                    values = [float(x[metric]) for x in grouped.get((condition, style), [])]
                    if values:
                        m, s = mean_std(values); cells.append(f"{m:.4f} ± {s:.4f}")
                    else:
                        cells.append("—")
                lines.append(f"| {condition} | " + " | ".join(cells) + " |")
            lines.append("")

    cirr = []
    for path in sorted(METRICS.glob("surface50k_cirr_*.json")):
        stem = path.stem.removeprefix("surface50k_cirr_")
        parts = stem.split("_")
        if len(parts) != 3 and len(parts) != 4:
            raise ValueError(f"unexpected CIRR metric filename: {path.name}")
        condition, seed = parts[:2]
        metric = "_".join(parts[2:])
        pred = json.loads(path.read_text())
        rows = []
        for item in GT:
            p = pred[str(item["pairid"])]
            try:
                rank = p.index(item["target_hard"]) + 1
            except ValueError:
                rank = 51 if metric == "recall" else 4
            rows.append(rank)
        max_k = (1, 5, 10, 50) if metric == "recall" else (1, 2, 3)
        row = {"condition": condition, "seed": int(seed[1:]), "metric": metric, "source": str(path), "n": len(rows)}
        for k in max_k:
            row[f"Recall@{k}"] = sum(r <= k for r in rows) / len(rows)
        row["mAP"] = sum((1 / r if r <= (50 if metric == "recall" else 3) else 0) for r in rows) / len(rows)
        cirr.append(row)
    cirr_fields = sorted({k for row in cirr for k in row}, key=lambda x: (x not in ("condition", "seed", "metric"), x))
    cirr_path = METRICS / "surface_control_cirr_metrics.csv"
    with cirr_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cirr_fields); w.writeheader(); w.writerows(cirr)

    lines += [f"CIRR metric files available: **{len(cirr)}/18**.", "", "| Condition | Metric | Recall@1 | Recall@5/2 | Recall@10/3 | mAP |", "|---|---|---:|---:|---:|---:|"]
    for row in cirr:
        k2, k3 = (("Recall@5", "Recall@10") if row["metric"] == "recall" else ("Recall@2", "Recall@3"))
        lines.append(f"| {row['condition']} | {row['metric']} | {row['Recall@1']:.4f} | {row[k2]:.4f} | {row[k3]:.4f} | {row['mAP']:.4f} |")
    lines += ["", "Interpretation should be written only after all matched cells are present. The Surface condition isolates lexical framing around the Fixed semantic core; it is not a semantic-intent label.", ""]
    (ROOT / "narrative_followup/reports/surface_control_report.md").write_text("\n".join(lines))
    print(json.dumps({"mtcir_files": len(mtcir), "cirr_files": len(cirr), "mtcir_csv": str(mtcir_path), "cirr_csv": str(cirr_path)}, indent=2))


if __name__ == "__main__":
    main()
