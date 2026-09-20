#!/usr/bin/env python3
"""Validate and summarize the two-annotator CIRR overlap sheet."""
from __future__ import annotations

import csv
import json
import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHEET = ROOT / "narrative_followup/annotations/cirr_mechanism_overlap_100.csv"
OUT = ROOT / "narrative_followup/annotations/annotation_agreement_v2.md"
ALLOWED_PRIMARY = {
    "attribute_style", "entity_replacement", "count", "spatial_relation",
    "negation_removal", "scene_context", "function_use_case",
    "comparative_intensity", "omission", "intent_drift", "entity_binding",
    "operation_polarity", "none", "ambiguous",
}
ALLOWED_GROUNDING = {"global", "local", "both", "unclear"}
ALLOWED_POLARITY = {"add", "remove", "replace", "move", "intensify", "reduce", "compare", "none", "unclear"}


def kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = sorted(set(left) | set(right))
    p_left = Counter(left)
    p_right = Counter(right)
    expected = sum((p_left[label] / len(left)) * (p_right[label] / len(left)) for label in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def summarize(field: str, allowed: set[str], rows: list[dict]) -> dict:
    left_key = f"annotator_1_{field}"
    right_key = f"annotator_2_{field}"
    left = [row[left_key].strip() for row in rows]
    right = [row[right_key].strip() for row in rows]
    invalid = sorted({value for value in left + right if value and value not in allowed})
    filled = sum(bool(a and b) for a, b in zip(left, right))
    return {
        "field": field,
        "filled_both": filled,
        "total": len(rows),
        "invalid": invalid,
        "agreement": sum(a == b for a, b in zip(left, right)) / filled if filled else None,
        "kappa": kappa([a for a, b in zip(left, right) if a and b], [b for a, b in zip(left, right) if a and b]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-human",
        action="store_true",
        help="Record that human annotation was intentionally skipped in this release.",
    )
    args = parser.parse_args()
    if args.skip_human:
        OUT.write_text(
            "# CIRR annotation agreement v2\n\n"
            "Status: **SKIPPED_NO_ANNOTATORS**.\n\n"
            "Human mechanism annotation and double-annotation agreement were "
            "intentionally skipped for this release because annotators were "
            "unavailable. The 600-row and 100-row sheets are retained as "
            "unfilled audit materials. No human-label or category-specific "
            "mechanism result is reported.\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(OUT), "status": "SKIPPED_NO_ANNOTATORS"}, indent=2))
        return

    with SHEET.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise SystemExit(f"expected 100 overlap rows, found {len(rows)}")
    summaries = [
        summarize("primary_mechanism", ALLOWED_PRIMARY, rows),
        summarize("grounding", ALLOWED_GROUNDING, rows),
        summarize("operation_polarity", ALLOWED_POLARITY, rows),
    ]
    pending = any(item["filled_both"] < 100 or item["invalid"] for item in summaries)
    lines = [
        "# CIRR annotation agreement v2", "",
        "This report is generated from the 100-row overlap sheet. It is not a human result until both annotator columns are filled.", "",
        "| Field | Both filled | Rows | Exact agreement | Cohen's kappa | Invalid values |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        agreement = "pending" if item["agreement"] is None else f"{item['agreement']:.3f}"
        kap = "pending" if item["kappa"] is None else f"{item['kappa']:.3f}"
        lines.append(f"| {item['field']} | {item['filled_both']} | {item['total']} | {agreement} | {kap} | {', '.join(item['invalid']) or 'none'} |")
    lines += ["", f"Status: **{'PENDING_HUMAN_LABELS' if pending else 'READY_FOR_REVIEW'}**.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "status": "PENDING_HUMAN_LABELS" if pending else "READY_FOR_REVIEW", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
