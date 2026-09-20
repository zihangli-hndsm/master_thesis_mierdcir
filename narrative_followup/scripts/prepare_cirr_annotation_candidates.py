#!/usr/bin/env python3
"""Create a disjoint, seed-audited CIRR mechanism annotation queue."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DELTA = ROOT / "narrative_followup/metrics/cirr_val_rank_deltas_aggregated.csv"
PER_SEED = ROOT / "narrative_followup/metrics/cirr_val_rank_deltas_per_seed.csv"
CIRR = ROOT / "data/CIRR/cap.rc2.val.json"
PRED = ROOT / "narrative_followup/predictions/cirr_val"
OUT = ROOT / "narrative_followup/annotations/cirr_mechanism_candidates.csv"
MANIFEST = ROOT / "narrative_followup/manifests/cirr_mechanism_selection_manifest.json"
SEEDS = ("42", "123", "2025")
RANDOM_SEED = 20260826


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(condition: str, seed: int, metric: str) -> dict[str, list[str]]:
    path = PRED / f"{condition}_s{seed}_{metric}.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): value for key, value in payload.items() if str(key).isdigit()}


def select_sorted(rows: list[dict], n: int, used: set[str], key, reverse: bool) -> list[dict]:
    chosen = []
    for row in sorted(rows, key=key, reverse=reverse):
        if row["pairid"] in used:
            continue
        used.add(row["pairid"])
        chosen.append(row)
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise RuntimeError(f"could not select {n} rows; selected {len(chosen)}")
    return chosen


def winner(delta: float) -> str:
    if delta > 0:
        return "multi"
    if delta < 0:
        return "fixed"
    return "tie"


def stratified_random(rows: list[dict], n: int, used: set[str]) -> list[dict]:
    """Deterministically sample round-robin across global/subset sign strata."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["pairid"] in used:
            continue
        g = "+" if row["delta_global"] > 0 else "-" if row["delta_global"] < 0 else "0"
        s = "+" if row["delta_subset"] > 0 else "-" if row["delta_subset"] < 0 else "0"
        buckets[f"global_{g}_subset_{s}"].append(row)
    rng = random.Random(RANDOM_SEED)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    names = sorted(buckets)
    chosen = []
    while len(chosen) < n and any(buckets[name] for name in names):
        for name in names:
            if not buckets[name]:
                continue
            row = buckets[name].pop()
            used.add(row["pairid"])
            chosen.append(row)
            if len(chosen) == n:
                return chosen
    raise RuntimeError(f"could not select {n} stratified rows; selected {len(chosen)}")


def main() -> None:
    with DELTA.open(encoding="utf-8", newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    with PER_SEED.open(encoding="utf-8", newline="") as handle:
        per_seed_rows = list(csv.DictReader(handle))
    by_pair_seed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in per_seed_rows:
        by_pair_seed[str(row["pairid"])][str(row["seed"])] = row
    for pairid, seed_rows in by_pair_seed.items():
        if set(seed_rows) != set(SEEDS):
            raise RuntimeError(f"pair {pairid} does not have exactly seeds {SEEDS}: {sorted(seed_rows)}")

    candidates = []
    for raw in aggregate_rows:
        pairid = str(raw["pairid"])
        seed_rows = by_pair_seed[pairid]
        global_deltas = [float(seed_rows[seed]["delta_global_fixed_minus_multi"]) for seed in SEEDS]
        subset_deltas = [float(seed_rows[seed]["delta_subset_fixed_minus_multi"]) for seed in SEEDS]
        candidates.append({
            "pairid": pairid,
            "reference": raw["reference"],
            "target": raw["target"],
            "delta_global": float(raw["delta_global_fixed_minus_multi"]),
            "delta_subset": float(raw["delta_subset_fixed_minus_multi"]),
            "global_rank_fixed": raw["fixed_global_rank_censored_51"],
            "global_rank_multi": raw["multi_global_rank_censored_51"],
            "subset_rank_fixed": raw["fixed_subset_rank_censored_4"],
            "subset_rank_multi": raw["multi_subset_rank_censored_4"],
            "seed_global_deltas": global_deltas,
            "seed_subset_deltas": subset_deltas,
            "global_positive_all_seeds": all(value > 0 for value in global_deltas),
            "global_negative_all_seeds": all(value < 0 for value in global_deltas),
            "subset_positive_all_seeds": all(value > 0 for value in subset_deltas),
            "subset_negative_all_seeds": all(value < 0 for value in subset_deltas),
        })

    used: set[str] = set()
    selected: list[tuple[str, str, dict]] = []
    # Positive global delta means Multi wins; negative subset delta means
    # Fixed wins within the subset, i.e. a Multi subset loss.
    subset_losses = select_sorted(
        [r for r in candidates if r["subset_negative_all_seeds"]],
        150,
        used,
        key=lambda r: (r["delta_subset"], r["pairid"]),
        reverse=False,
    )
    selected.extend(("multi_subset_loss", "all_three_subset_deltas_negative", row) for row in subset_losses)

    global_wins = select_sorted(
        [r for r in candidates if r["global_positive_all_seeds"]],
        150,
        used,
        key=lambda r: (-r["delta_global"], r["pairid"]),
        reverse=False,
    )
    selected.extend(("multi_global_win", "all_three_global_deltas_positive", row) for row in global_wins)

    fixed_wins = select_sorted(
        [r for r in candidates if r["global_negative_all_seeds"]],
        100,
        used,
        key=lambda r: (r["delta_global"], r["pairid"]),
        reverse=False,
    )
    selected.extend(("fixed_win", "all_three_global_deltas_negative", row) for row in fixed_wins)

    random_rows = stratified_random(candidates, 200, used)
    selected.extend(("stratified_random", "round_robin_sign_strata_seed_20260826", row) for row in random_rows)

    with CIRR.open(encoding="utf-8") as handle:
        queries = {str(row["pairid"]): row for row in json.load(handle)}
    fixed_global = load_predictions("fixed", 42, "recall")
    multi_global = load_predictions("multi", 42, "recall")
    fixed_subset = load_predictions("fixed", 42, "recall_subset")
    multi_subset = load_predictions("multi", 42, "recall_subset")

    fields = [
        "annotation_group", "selection_rule", "pairid", "reference", "target", "caption",
        "delta_global_fixed_minus_multi", "delta_subset_fixed_minus_multi",
        "fixed_global_rank", "multi_global_rank", "fixed_subset_rank", "multi_subset_rank",
        "seed_global_deltas", "seed_subset_deltas",
        "global_positive_all_seeds", "global_negative_all_seeds",
        "subset_positive_all_seeds", "subset_negative_all_seeds",
        "fixed_global_top5", "multi_global_top5", "fixed_subset_top4", "multi_subset_top4",
        "primary_mechanism", "additional_tags", "grounding", "operation_polarity",
        "winner_global", "winner_subset", "annotator", "notes",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, selection_rule, row in selected:
            query = queries.get(row["pairid"], {})
            g_fixed = fixed_global.get(row["pairid"], [])
            g_multi = multi_global.get(row["pairid"], [])
            s_fixed = fixed_subset.get(row["pairid"], [])
            s_multi = multi_subset.get(row["pairid"], [])
            writer.writerow({
                "annotation_group": group,
                "selection_rule": selection_rule,
                "pairid": row["pairid"],
                "reference": row["reference"],
                "target": row["target"],
                "caption": query.get("caption", ""),
                "delta_global_fixed_minus_multi": row["delta_global"],
                "delta_subset_fixed_minus_multi": row["delta_subset"],
                "fixed_global_rank": row["global_rank_fixed"],
                "multi_global_rank": row["global_rank_multi"],
                "fixed_subset_rank": row["subset_rank_fixed"],
                "multi_subset_rank": row["subset_rank_multi"],
                "seed_global_deltas": "|".join(f"{value:.6g}" for value in row["seed_global_deltas"]),
                "seed_subset_deltas": "|".join(f"{value:.6g}" for value in row["seed_subset_deltas"]),
                "global_positive_all_seeds": row["global_positive_all_seeds"],
                "global_negative_all_seeds": row["global_negative_all_seeds"],
                "subset_positive_all_seeds": row["subset_positive_all_seeds"],
                "subset_negative_all_seeds": row["subset_negative_all_seeds"],
                "fixed_global_top5": " | ".join(g_fixed[:5]),
                "multi_global_top5": " | ".join(g_multi[:5]),
                "fixed_subset_top4": " | ".join(s_fixed[:4]),
                "multi_subset_top4": " | ".join(s_multi[:4]),
                "primary_mechanism": "",
                "additional_tags": "",
                "grounding": "",
                "operation_polarity": "",
                "winner_global": winner(row["delta_global"]),
                "winner_subset": winner(row["delta_subset"]),
                "annotator": "",
                "notes": "",
            })

    group_counts = Counter(group for group, _, _ in selected)
    manifest = {
        "status": "PASS" if len(selected) == 600 and len(used) == 600 else "FAIL",
        "source_aggregated": str(DELTA),
        "source_aggregated_sha256": sha256(DELTA),
        "source_per_seed": str(PER_SEED),
        "source_per_seed_sha256": sha256(PER_SEED),
        "seed_ids": list(SEEDS),
        "selection_seed": RANDOM_SEED,
        "direction_convention": {
            "positive_global_delta": "Multi target rank earlier",
            "negative_subset_delta": "Fixed target rank earlier within subset; Multi subset loss",
        },
        "group_rules": {
            "multi_subset_loss": "150 unique pairs with delta_subset < 0 for all three seeds",
            "multi_global_win": "150 unique pairs with delta_global > 0 for all three seeds",
            "fixed_win": "100 unique pairs with delta_global < 0 for all three seeds",
            "stratified_random": "200 remaining unique pairs sampled round-robin over global/subset sign strata",
        },
        "group_counts": dict(group_counts),
        "total_rows": len(selected),
        "unique_pairids": len(used),
        "human_labels_present": False,
        "selected_pairids": [row["pairid"] for _, _, row in selected],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "manifest": str(MANIFEST), "rows": len(selected), "groups": dict(group_counts)}, indent=2))


if __name__ == "__main__":
    main()
