#!/usr/bin/env python3
"""Automatic CIRR rank-delta, text-feature, and extreme-case analysis.

The text features are deliberately simple proxies.  They are useful for
stratification and annotation triage, not for assigning a mechanism or
claiming a causal explanation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RANKS = ROOT / "narrative_followup/metrics/cirr_val_rank_deltas_per_seed.csv"
DEFAULT_GT = ROOT / "data/CIRR/cap.rc2.val.json"
OUT_DIR = ROOT / "narrative_followup/metrics"
REPORT = ROOT / "narrative_followup/reports/cirr_mechanism_auto_analysis_v2.md"


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            result[order[k]] = rank
        i = j + 1
    return result


def correlation(xs: list[float], ys: list[float]) -> tuple[float, float]:
    return pearson(xs, ys), pearson(ranks(xs), ranks(ys))


def text_features(text: str) -> dict[str, int]:
    low = text.lower()
    words = re.findall(r"\b[\w'-]+\b", low)
    # This is a transparent determiner/number-led noun proxy, not a parsed
    # entity count.  It is intentionally named as a proxy in every output.
    determiner = r"(?:a|an|the|another|one|two|three|four|five|six|several|multiple|each|both)"
    entity_proxy = len(re.findall(rf"\b{determiner}\s+[a-z][\w-]*", low))
    negation = bool(re.search(
        r"\b(no|not|never|without|remove|removed|eliminate|delete|erase|leave only|keep only|all but)\b",
        low,
    ))
    binding = bool(re.search(r"\b(it|them|this|that|these|those|the one|the other|ones)\b", low))
    return {
        "word_count": len(words),
        "char_count": len(text),
        "entity_mention_proxy_count": entity_proxy,
        "negation_proxy": int(negation),
        "entity_binding_proxy": int(binding),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.4f}"


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=Path, default=DEFAULT_RANKS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    args = ap.parse_args()

    gt = {str(row["pairid"]): row for row in json.loads(args.gt.read_text(encoding="utf-8"))}
    with args.ranks.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        pairid = str(raw["pairid"])
        if pairid not in gt:
            raise KeyError(f"pairid {pairid} missing from {args.gt}")
        caption = str(gt[pairid].get("caption", ""))
        features = text_features(caption)
        delta_global = float(raw["delta_global_fixed_minus_multi"])
        delta_subset = float(raw["delta_subset_fixed_minus_multi"])
        rows.append({
            "pairid": pairid,
            "reference": raw["reference"],
            "target": raw["target"],
            "seed": raw["seed"],
            "caption": caption,
            "fixed_global_rank_censored_51": raw["fixed_global_rank_censored_51"],
            "multi_global_rank_censored_51": raw["multi_global_rank_censored_51"],
            "delta_global_fixed_minus_multi": delta_global,
            "fixed_subset_rank_censored_4": raw["fixed_subset_rank_censored_4"],
            "multi_subset_rank_censored_4": raw["multi_subset_rank_censored_4"],
            "delta_subset_fixed_minus_multi": delta_subset,
            "global_multi_win": int(delta_global > 0),
            "global_fixed_win": int(delta_global < 0),
            "subset_multi_win": int(delta_subset > 0),
            "subset_fixed_win": int(delta_subset < 0),
            "global_win_subset_loss": int(delta_global > 0 and delta_subset < 0),
            **features,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_fields = list(rows[0])
    per_seed_path = OUT_DIR / "cirr_tradeoff_features_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows(rows)

    # Summaries by outcome group, retaining the automatic-proxy boundary.
    groups = {
        "all": rows,
        "global_multi_win": [r for r in rows if r["global_multi_win"]],
        "global_fixed_win": [r for r in rows if r["global_fixed_win"]],
        "subset_multi_win": [r for r in rows if r["subset_multi_win"]],
        "subset_fixed_win": [r for r in rows if r["subset_fixed_win"]],
        "global_win_subset_loss": [r for r in rows if r["global_win_subset_loss"]],
    }
    feature_names = [
        "word_count",
        "char_count",
        "entity_mention_proxy_count",
        "negation_proxy",
        "entity_binding_proxy",
    ]
    summary_fields = ["group", "n", *[f"mean_{name}" for name in feature_names]]
    summary_rows = []
    for name, group in groups.items():
        summary_rows.append({
            "group": name,
            "n": len(group),
            **{f"mean_{feature}": f"{mean([float(r[feature]) for r in group]):.6f}" for feature in feature_names},
        })
    summary_path = OUT_DIR / "cirr_tradeoff_text_feature_summary_v2.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    global_delta = [float(r["delta_global_fixed_minus_multi"]) for r in rows]
    subset_delta = [float(r["delta_subset_fixed_minus_multi"]) for r in rows]
    binary_global = [float(r["global_multi_win"]) for r in rows]
    binary_subset_loss = [float(r["subset_fixed_win"]) for r in rows]
    corr_rows = [
        {
            "comparison": "continuous_delta_global_vs_delta_subset",
            "n": len(rows),
            "pearson_r": fmt(pearson(global_delta, subset_delta)),
            "spearman_r": fmt(correlation(global_delta, subset_delta)[1]),
            "definition": "delta_global=fixed_rank-minus-multi_rank; delta_subset=same sign convention",
        },
        {
            "comparison": "global_improvement_vs_subset_degradation",
            "n": len(rows),
            "pearson_r": fmt(pearson(binary_global, binary_subset_loss)),
            "spearman_r": fmt(correlation(binary_global, binary_subset_loss)[1]),
            "definition": "binary global improvement=(delta_global>0); subset degradation=(delta_subset<0)",
        },
    ]
    corr_path = OUT_DIR / "cirr_tradeoff_correlations_v2.csv"
    with corr_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
        writer.writeheader()
        writer.writerows(corr_rows)

    # Query-level paired bootstrap: all three seed observations for a pair
    # are resampled together, so seed repetition is not treated as 12,543
    # independent queries.
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pairid"])].append(row)
    pair_summaries = []
    for pairid, pair_rows in sorted(by_pair.items()):
        if len(pair_rows) != 3:
            raise RuntimeError(f"expected three seed rows for {pairid}, got {len(pair_rows)}")
        pair_summaries.append({
            "global_delta_mean": mean([float(r["delta_global_fixed_minus_multi"]) for r in pair_rows]),
            "subset_delta_mean": mean([float(r["delta_subset_fixed_minus_multi"]) for r in pair_rows]),
            "global_multi_win_rate": mean([float(r["global_multi_win"]) for r in pair_rows]),
            "subset_fixed_win_rate": mean([float(r["subset_fixed_win"]) for r in pair_rows]),
            "global_win_subset_loss_rate": mean([float(r["global_win_subset_loss"]) for r in pair_rows]),
        })
    bootstrap_rng = random.Random(20260826)
    bootstrap_replicates = 2000
    metric_names = list(pair_summaries[0])
    bootstrap_values = {name: [] for name in metric_names}
    for _ in range(bootstrap_replicates):
        sample = [pair_summaries[bootstrap_rng.randrange(len(pair_summaries))] for _ in pair_summaries]
        for name in metric_names:
            bootstrap_values[name].append(mean([float(item[name]) for item in sample]))
    observed = {name: mean([float(item[name]) for item in pair_summaries]) for name in metric_names}
    bootstrap_summary = {
        "unit": "CIRR pairid; all three seed observations resampled together",
        "n_pairs": len(pair_summaries),
        "n_seed_observations": len(rows),
        "replicates": bootstrap_replicates,
        "random_seed": 20260826,
        "confidence_level": 0.95,
        "metrics": {
            name: {
                "observed": observed[name],
                "ci_lower": percentile(values, 0.025),
                "ci_upper": percentile(values, 0.975),
            }
            for name, values in bootstrap_values.items()
        },
        "automatic_only": True,
    }
    bootstrap_path = OUT_DIR / "cirr_tradeoff_bootstrap_ci_v2.json"
    bootstrap_path.write_text(json.dumps(bootstrap_summary, indent=2) + "\n", encoding="utf-8")

    # Export a compact, reproducible set of extreme rows for annotation triage.
    extreme_specs = [
        ("global_multi_win", lambda r: float(r["delta_global_fixed_minus_multi"]), True),
        ("global_fixed_win", lambda r: float(r["delta_global_fixed_minus_multi"]), False),
        ("subset_multi_win", lambda r: float(r["delta_subset_fixed_minus_multi"]), True),
        ("subset_fixed_win", lambda r: float(r["delta_subset_fixed_minus_multi"]), False),
    ]
    extreme_rows = []
    for label, key, reverse in extreme_specs:
        chosen = []
        seen_pairids: set[str] = set()
        for candidate in sorted(rows, key=key, reverse=reverse):
            # Keep the strongest seed-level example for each query so the
            # handoff does not show the same visual pair three times.
            if str(candidate["pairid"]) in seen_pairids:
                continue
            seen_pairids.add(str(candidate["pairid"]))
            chosen.append(candidate)
            if len(chosen) == 10:
                break
        for rank, row in enumerate(chosen, start=1):
            extreme_rows.append({"extreme_type": label, "extreme_rank": rank, **row})
    extreme_path = OUT_DIR / "cirr_tradeoff_extremes_v2.csv"
    with extreme_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["extreme_type", "extreme_rank", *feature_fields])
        writer.writeheader()
        writer.writerows(extreme_rows)

    # Markdown report.
    lines = [
        "# CIRR automatic mechanism-oriented analysis v2",
        "",
        "This is an automatic stratification and annotation-triage report. The",
        "text fields are lexical proxies, not entity parsing or human mechanism",
        "labels. No category-level causal claim is supported by this report.",
        "",
        "## Rank-delta relationship",
        "",
        f"Rows: **{len(rows)}** query-seed observations (4,181 queries × 3 seeds).",
        "Positive delta means Multi places the target earlier under either mode.",
        "For the binary analysis, global improvement is `delta_global > 0` and",
        "subset degradation is `delta_subset < 0` (Fixed places the target earlier",
        "within the subset).",
        "",
        "| Comparison | n | Pearson r | Spearman r |",
        "|---|---:|---:|---:|",
    ]
    for row in corr_rows:
        lines.append(f"| {row['comparison']} | {row['n']} | {row['pearson_r']} | {row['spearman_r']} |")
    lines.extend([
        "",
        "## Outcome counts and text proxies",
        "",
        "`entity_mention_proxy_count` counts determiner/number-led lexical spans;",
        "it is not a named-entity or noun-phrase parser. `negation_proxy` and",
        "`entity_binding_proxy` are binary lexical indicators.",
        "",
        "| Group | n | mean words | mean chars | mean entity proxy | negation proxy | binding proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {float(row['mean_word_count']):.2f} | "
            f"{float(row['mean_char_count']):.2f} | {float(row['mean_entity_mention_proxy_count']):.2f} | "
            f"{float(row['mean_negation_proxy']):.4f} | {float(row['mean_entity_binding_proxy']):.4f} |"
        )
    lines.extend([
        "",
        "## Extreme rows for human review",
        "",
        "The full rows are in `cirr_tradeoff_extremes_v2.csv`, with captions and",
        "frozen ranks. Each extreme type contains unique pair IDs and keeps the",
        "most extreme seed-level observation for each pair. These are triage samples only; reference/target contact",
        "sheets and human labels remain authoritative.",
        "",
        "| Extreme type | Rows exported | Selection rule |",
        "|---|---:|---|",
        "| global_multi_win | 10 | largest positive global delta |",
        "| global_fixed_win | 10 | most negative global delta |",
        "| subset_multi_win | 10 | largest positive subset delta |",
        "| subset_fixed_win | 10 | most negative subset delta |",
        "",
        "Machine-readable outputs:",
        "",
        "- `cirr_tradeoff_features_per_seed.csv`",
        "- `cirr_tradeoff_text_feature_summary_v2.csv`",
        "- `cirr_tradeoff_correlations_v2.csv`",
        "- `cirr_tradeoff_extremes_v2.csv`",
        "- `cirr_tradeoff_bootstrap_ci_v2.json`",
        "",
        "The bootstrap is paired at the query/pair level: the three seed rows",
        "for each pair are resampled together. It quantifies query-level",
        "uncertainty and does not replace separate seed variability or human",
        "annotation.",
        "",
        "All outputs are automatic-only and must not be used to replace the",
        "planned human mechanism annotation and double-annotation agreement.",
        "",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={per_seed_path} rows={len(rows)}")
    print(f"wrote={summary_path} rows={len(summary_rows)}")
    print(f"wrote={corr_path} rows={len(corr_rows)}")
    print(f"wrote={extreme_path} rows={len(extreme_rows)}")
    print(f"wrote={bootstrap_path} replicates={bootstrap_replicates}")
    print(f"wrote={REPORT}")
    print("status=automatic_proxy_only; human_visual_review_required=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
