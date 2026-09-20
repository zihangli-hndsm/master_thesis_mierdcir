#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/home/zihali/final_thesis")
OUT = ROOT / "narrative_followup"


def main() -> None:
    records = []
    query_files = sorted(
        p for p in (OUT / "metrics").glob("query_*.json")
        if p.name.startswith(("query_raw_", "query_fixed_", "query_multi_"))
    )
    for p in query_files:
        # query_<condition>_s<seed>_<style>.json
        stem = p.stem.removeprefix("query_")
        cond, seed, style = stem.rsplit("_", 2)
        data = json.loads(p.read_text())
        row = {"condition": cond, "seed": int(seed[1:]), "query_style": style, "source": str(p)}
        row.update(data["metrics"])
        records.append(row)
    assert len(records) == 27, f"expected 27 matrix cells, got {len(records)}"

    fields = ["condition", "seed", "query_style", "Recall@1", "Recall@5", "Recall@10", "Recall@50", "mAP", "source"]
    for output_name in ("query_style_matrix.csv", "query_style_matrix_per_seed.csv"):
        with (OUT / f"metrics/{output_name}").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)

    grouped = {}
    for r in records:
        grouped.setdefault((r["condition"], r["query_style"]), []).append(r)
    metrics = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "mAP"]
    summary = {}
    for key, rows in grouped.items():
        summary[key] = {}
        for metric in metrics:
            vals = [float(r[metric]) for r in rows]
            summary[key][metric] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals)}

    lines = [
        "# Query-style alignment matrix",
        "",
        "This is a post-hoc evaluation of the nine frozen pair-control checkpoints on the same 10,000 pair-held-out MTCIR test pairs. Only the modification text changes across Raw, Fixed and Multi query styles; reference images, target images, gallery and checkpoint are fixed.",
        "",
        "All values below are mean ± sample standard deviation over seeds 42, 123 and 2025.",
        "",
    ]
    for metric in metrics:
        lines += [f"## {metric}", "", "| train condition \\ query style | Raw | Fixed | Multi |", "|---|---:|---:|---:|"]
        for cond in ("raw", "fixed", "multi"):
            vals=[]
            for style in ("raw", "fixed", "multi"):
                x=summary[(cond,style)][metric]
                vals.append(f"{x['mean']:.4f} ± {x['std']:.4f}")
            lines.append(f"| {cond} | " + " | ".join(vals) + " |")
        lines.append("")

    lines += [
        "## Interpretation rule",
        "",
        "A matched-style diagonal advantage supports query-style alignment. A higher row average across all three styles supports broader style coverage. This matrix is descriptive and does not by itself isolate semantic intent diversity from surface-form differences.",
        "",
        "The raw per-cell values are in `metrics/query_style_matrix.csv` and "
        "`metrics/query_style_matrix_per_seed.csv`; the 270,000-row target-rank "
        "audit is in `metrics/query_style_ranks.csv` with its validation report "
        "in `reports/query_style_rank_audit.md`. The input alignment audit is in "
        "`manifests/query_style_alignment.json`.",
    ]
    (OUT / "reports/query_style_matrix_report.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        for metric in ("Recall@1", "Recall@10", "mAP"):
            arr = np.array([[summary[(c,s)][metric]["mean"] for s in ("raw","fixed","multi")] for c in ("raw","fixed","multi")])
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im=ax.imshow(arr, cmap="viridis")
            ax.set_xticks(range(3), ["Raw queries","Fixed queries","Multi queries"], rotation=25, ha="right")
            ax.set_yticks(range(3), ["Raw model","Fixed model","Multi model"])
            ax.set_title(f"Query-style matrix: {metric}")
            for i in range(3):
                for j in range(3): ax.text(j,i,f"{arr[i,j]:.3f}",ha="center",va="center",color="white" if arr[i,j] < arr.mean() else "black")
            fig.colorbar(im, ax=ax, label=metric); fig.tight_layout()
            fig.savefig(OUT / f"figures/query_style_heatmap_{metric.replace('@','at').replace('.','')}.pdf")
            plt.close(fig)
    except Exception as exc:
        (OUT / "reports/query_style_heatmap_error.txt").write_text(repr(exc) + "\n")

    print(f"matrix records={len(records)}")


if __name__ == "__main__":
    main()
