#!/usr/bin/env python3
"""Prepare a balanced 100-row double-annotation overlap sheet."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "narrative_followup/annotations/cirr_mechanism_candidates.csv"
OUT = ROOT / "narrative_followup/annotations/cirr_mechanism_overlap_100.csv"
GROUPS = ("multi_global_win", "multi_subset_loss", "fixed_win", "stratified_random")


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for group in GROUPS:
        group_rows = [row for row in rows if row["annotation_group"] == group]
        if len(group_rows) < 25:
            raise SystemExit(f"need at least 25 rows for {group}, found {len(group_rows)}")
        selected.extend(group_rows[:25])
    fields = [
        "annotation_group", "pairid", "reference", "target", "caption",
        "delta_global_fixed_minus_multi", "delta_subset_fixed_minus_multi",
        "fixed_global_rank", "multi_global_rank", "fixed_subset_rank", "multi_subset_rank",
        "fixed_global_top5", "multi_global_top5", "fixed_subset_top4", "multi_subset_top4",
        "winner_global", "winner_subset",
        "annotator_1_primary_mechanism", "annotator_1_additional_tags",
        "annotator_1_grounding", "annotator_1_operation_polarity", "annotator_1_notes",
        "annotator_2_primary_mechanism", "annotator_2_additional_tags",
        "annotator_2_grounding", "annotator_2_operation_polarity", "annotator_2_notes",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fields})
    print({"output": str(OUT), "rows": len(selected), "groups": dict(Counter(row["annotation_group"] for row in selected))})


if __name__ == "__main__":
    main()
