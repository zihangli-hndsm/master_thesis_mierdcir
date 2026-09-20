#!/usr/bin/env python3
"""Summarize automatic intent distributions and descriptive CIRR associations.

All labels in this analysis are lexical automatic labels.  The output is
descriptive and is intentionally separate from the human-review columns.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = [
    "negative_constraint",
    "spatial_positional",
    "comparative_intensity",
    "functional_affordance",
    "global_view",
    "instance_level",
]
SOURCE_ORDER = ["MTCIR_test_raw", "MerdCIR_test", "CIRR_val", "FashionIQ_val"]


def distribution(counts: Counter[str]) -> list[float]:
    total = sum(counts.values())
    return [counts[label] / total if total else 0.0 for label in LABELS]


def kl_bits(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def js_bits(p: list[float], q: list[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]
    return (kl_bits(p, midpoint) + kl_bits(q, midpoint)) / 2.0


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    labels_path = ROOT / "narrative_followup/metrics/intent_query_labels_auto_v2.csv"
    coverage_path = ROOT / "narrative_followup/metrics/intent_coverage_auto_v2.csv"
    performance_path = ROOT / "narrative_followup/metrics/intent_performance_cirr_v2.csv"
    out_dir = ROOT / "narrative_followup/metrics"
    report_path = ROOT / "narrative_followup/reports/intent_distribution_analysis_v2.md"

    counts: dict[str, Counter[str]] = {source: Counter() for source in SOURCE_ORDER}
    with labels_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            if source not in counts:
                counts[source] = Counter()
            counts[source][row["auto_primary_intent"]] += 1
    proportions = {source: distribution(counter) for source, counter in counts.items()}
    uniform = [1.0 / len(LABELS)] * len(LABELS)

    dist_out = out_dir / "intent_distribution_js_v2.csv"
    with dist_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["left_source", "right_source", "js_divergence_bits"])
        writer.writeheader()
        for left in SOURCE_ORDER:
            for right in SOURCE_ORDER:
                writer.writerow({
                    "left_source": left,
                    "right_source": right,
                    "js_divergence_bits": f"{js_bits(proportions[left], proportions[right]):.10f}",
                })

    frequency_out = out_dir / "intent_frequency_performance_correlation_v2.csv"
    with performance_path.open(newline="", encoding="utf-8") as f:
        perf_rows = list(csv.DictReader(f))
    perf_by_key = {(r["condition"], r["metric"], r["intent"]): float(r["hit_at_1"]) for r in perf_rows}
    cirr_freq = proportions["CIRR_val"]
    corr_rows = []
    for condition in ("fixed", "multi"):
        for metric in ("recall", "recall_subset"):
            ys = [perf_by_key[(condition, metric, label)] for label in LABELS]
            corr_rows.append({
                "condition": condition,
                "metric": metric,
                "frequency_source": "CIRR_val",
                "n_intents": len(LABELS),
                "pearson_r": fmt(pearson(cirr_freq, ys)),
                "spearman_r": fmt(pearson(ranks(cirr_freq), ranks(ys))),
                "automatic_only": "True",
            })
    with frequency_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
        writer.writeheader()
        writer.writerows(corr_rows)

    lines = [
        "# Automatic intent distribution analysis v2",
        "",
        "This report uses the same six-label lexical taxonomy as the automatic",
        "coverage scan. It is descriptive only; the 400-row manual review sheet",
        "is still required before category-level or causal claims.",
        "",
        "## Benchmark distributions",
        "",
        "| Source | Rows | " + " | ".join(LABELS) + " | JS to uniform (bits) |",
        "|---|---:|" + "---:|" * (len(LABELS) + 1),
    ]
    for source in SOURCE_ORDER:
        counter = counts[source]
        values = proportions[source]
        lines.append(
            f"| {source} | {sum(counter.values())} | "
            + " | ".join(f"{v:.4f}" for v in values)
            + f" | {js_bits(values, uniform):.6f} |"
        )

    lines.extend([
        "",
        "## Pairwise Jensen–Shannon divergence (bits)",
        "",
        "| Left | Right | JS divergence |",
        "|---|---|---:|",
    ])
    for left in SOURCE_ORDER:
        for right in SOURCE_ORDER:
            lines.append(f"| {left} | {right} | {js_bits(proportions[left], proportions[right]):.6f} |")

    lines.extend([
        "",
        "## CIRR automatic intent performance",
        "",
        "Per-intent hit rates are copied from `intent_performance_cirr_v2.csv`;",
        "each rate uses automatic labels and is not human validated.",
        "",
        "| Condition | Metric | Intent | n | Hit@1 |",
        "|---|---|---|---:|---:|",
    ])
    for row in perf_rows:
        lines.append(f"| {row['condition']} | {row['metric']} | {row['intent']} | {row['n']} | {float(row['hit_at_1']):.4f} |")

    lines.extend([
        "",
        "## Frequency–performance association",
        "",
        "The correlation uses the six CIRR-val automatic intent proportions as",
        "frequency and the corresponding six per-intent CIRR hit rates as performance.",
        "With six points, these coefficients are exploratory summaries, not tests",
        "of causality.",
        "",
        "| Condition | Metric | Frequency source | Pearson r | Spearman r |",
        "|---|---|---|---:|---:|",
    ])
    for row in corr_rows:
        lines.append(f"| {row['condition']} | {row['metric']} | {row['frequency_source']} | {row['pearson_r']} | {row['spearman_r']} |")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={dist_out} rows={len(SOURCE_ORDER) ** 2}")
    print(f"wrote={frequency_out} rows={len(corr_rows)}")
    print(f"wrote={report_path}")
    print("status=automatic_descriptive_only; manual_validation_required=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
