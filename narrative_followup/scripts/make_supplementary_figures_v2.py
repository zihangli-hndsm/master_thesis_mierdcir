#!/usr/bin/env python3
"""Render automatic-only supplementary figures from frozen CSV/JSON outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ROOT / "narrative_followup/metrics"
FIGURES = ROOT / "narrative_followup/figures"
LABELS = [
    "negative_constraint",
    "spatial_positional",
    "comparative_intensity",
    "functional_affordance",
    "global_view",
    "instance_level",
]


def load_rank_rows() -> list[dict]:
    with (METRICS / "cirr_val_rank_deltas_per_seed.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["delta_global"] = float(row["delta_global_fixed_minus_multi"])
        row["delta_subset"] = float(row["delta_subset_fixed_minus_multi"])
    return rows


def figure_tradeoff(rows: list[dict]) -> None:
    x = np.asarray([r["delta_global"] for r in rows])
    y = np.asarray([r["delta_subset"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.6, 5.3))
    ax.scatter(x, y, s=4, alpha=0.08, color="#315a8a", rasterized=True)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Global delta: rank(Fixed) − rank(Multi)")
    ax.set_ylabel("Subset delta: rank(Fixed) − rank(Multi)")
    ax.set_title("CIRR global/subset rank-delta relationship")
    ax.text(
        0.02,
        0.98,
        "Positive global delta = Multi global improvement\n"
        "Negative subset delta = Multi subset loss\n"
        "Automatic rank evidence; not a causal mechanism result",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "global_subset_tradeoff.pdf", metadata={"Creator": "make_supplementary_figures_v2.py"})
    plt.close(fig)


def figure_enrichment(rows: list[dict]) -> None:
    labels_path = METRICS / "intent_query_labels_auto_v2.csv"
    with labels_path.open(newline="", encoding="utf-8") as f:
        labels = {
            str(r["id"]): r["auto_primary_intent"]
            for r in csv.DictReader(f)
            if r["source"] == "CIRR_val"
        }
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pairid"])].append(row)
    events = defaultdict(lambda: {"n": 0, "global": 0.0, "subset_loss": 0.0})
    overall = {"n": 0, "global": 0.0, "subset_loss": 0.0}
    for pairid, pair_rows in by_pair.items():
        label = labels.get(pairid)
        if label not in LABELS:
            continue
        global_rate = sum(r["delta_global"] > 0 for r in pair_rows) / len(pair_rows)
        subset_loss_rate = sum(r["delta_subset"] < 0 for r in pair_rows) / len(pair_rows)
        events[label]["n"] += 1
        events[label]["global"] += global_rate
        events[label]["subset_loss"] += subset_loss_rate
        overall["n"] += 1
        overall["global"] += global_rate
        overall["subset_loss"] += subset_loss_rate
    overall_global = overall["global"] / overall["n"]
    overall_subset = overall["subset_loss"] / overall["n"]
    global_ratio = [events[label]["global"] / events[label]["n"] / overall_global for label in LABELS]
    subset_ratio = [events[label]["subset_loss"] / events[label]["n"] / overall_subset for label in LABELS]

    positions = np.arange(len(LABELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(positions - width / 2, global_ratio, width, label="Multi global win", color="#3973a8")
    ax.bar(positions + width / 2, subset_ratio, width, label="Multi subset loss", color="#c77c30")
    ax.axhline(1.0, color="#555555", linewidth=0.8)
    ax.set_xticks(positions, [label.replace("_", "\n") for label in LABELS], fontsize=8)
    ax.set_ylabel("Rate / overall CIRR rate")
    ax.set_title("Automatic intent win/loss enrichment on CIRR val")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        "Labels are lexical automatic classifications; ratios are descriptive only.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "intent_win_loss_enrichment.pdf", metadata={"Creator": "make_supplementary_figures_v2.py"})
    plt.close(fig)


def figure_distributions() -> None:
    training = json.loads((METRICS / "intent_category_metrics.json").read_text(encoding="utf-8"))
    scenario_counts = training["labeled_exploratory_vlm_corpus"]["scenario_counts"]
    scenario_names = sorted(scenario_counts)
    scenario_values = np.asarray([scenario_counts[name] / sum(scenario_counts.values()) for name in scenario_names])

    benchmark = defaultdict(dict)
    with (METRICS / "intent_coverage_auto_v2.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            benchmark[row["source"]][row["intent"]] = float(row["proportion"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1, 1.7]})
    left.bar(np.arange(len(scenario_names)), scenario_values, color="#6a9f58")
    left.set_xticks(np.arange(len(scenario_names)), [name.replace("surface_diverse_", "S") for name in scenario_names], fontsize=8)
    left.set_ylabel("Proportion")
    left.set_title("Training scenario coverage")
    left.text(0.02, 0.02, "Separate scenario taxonomy", transform=left.transAxes, fontsize=8)

    for source in ("MTCIR_test_raw", "MerdCIR_test", "CIRR_val", "FashionIQ_val"):
        right.plot(LABELS, [benchmark[source].get(label, 0.0) for label in LABELS], marker="o", label=source)
    right.set_xticks(np.arange(len(LABELS)), [label.replace("_", "\n") for label in LABELS], fontsize=8)
    right.set_ylabel("Proportion")
    right.set_title("Benchmark automatic taxonomy")
    right.legend(frameon=False, fontsize=8)
    fig.suptitle("Automatic train/test distribution overview (taxonomies are not equivalent)", fontsize=12)
    fig.text(0.5, 0.01, "Training scenario labels and benchmark lexical labels must not be interpreted as a shared semantic ground truth.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIGURES / "train_test_intent_distribution.pdf", metadata={"Creator": "make_supplementary_figures_v2.py"})
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rank_rows()
    figure_tradeoff(rows)
    figure_enrichment(rows)
    figure_distributions()
    print("wrote=global_subset_tradeoff.pdf")
    print("wrote=intent_win_loss_enrichment.pdf")
    print("wrote=train_test_intent_distribution.pdf")
    print("status=automatic_only; semantic_human_validation_required=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
