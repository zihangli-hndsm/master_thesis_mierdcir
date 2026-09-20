#!/usr/bin/env python3
"""Create conservative, caption-only mechanism suggestions for human review.

This file intentionally writes a separate preannotation artifact.  The
suggestions are not human labels, are not used for agreement or paper
statistics, and do not replace visual inspection of the reference/target
contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RULES: list[tuple[str, str, str]] = [
    ("entity_replacement", r"\b(change|replace|switch|swap|turn|transform|convert)\b", "replacement_verb"),
    ("entity_replacement", r"\b(into|to)\s+(?:a|an|the)?\s*[a-z]+", "replacement_target"),
    ("count", r"\b(one|single|two|three|four|five|six|several|multiple|pair|both|each|fewer|more)\b", "count_word"),
    ("count", r"\b(all but|at least|no more than|only one|only two)\b", "count_phrase"),
    ("spatial_relation", r"\b(left|right|above|below|over|under|behind|front|foreground|background|next to|beside|near|far|opposite|direction|facing)\b", "spatial_word"),
    ("spatial_relation", r"\b(move|put|place|position)\b", "spatial_verb"),
    ("negation_removal", r"\b(remove|removed|without|no longer|eliminate|get rid of|delete|erase)\b", "removal_word"),
    ("negation_removal", r"\b(leave only|all but|keep only)\b", "restrictive_removal"),
    ("scene_context", r"\b(background|scene|setting|forest|beach|room|outdoor|indoors|street|city|mountain|water|sky|grass|snow|coast)\b", "scene_word"),
    ("function_use_case", r"\b(use|用途|wear|worn|hold|carry|riding|driving|playing|working|cooking|sleeping)\b", "function_word"),
    ("comparative_intensity", r"\b(more|less|bigger|larger|smaller|darker|lighter|brighter|stronger|weaker|thicker|thinner|taller|shorter|increase|decrease|intensif)\b", "comparative_word"),
    ("attribute_style", r"\b(colou?r|red|blue|green|black|white|yellow|orange|pink|purple|brown|gray|grey|dark|light|bright|pattern|striped|plain|style|design|material|texture|shape)\b", "attribute_word"),
    ("omission", r"\b(only|just|leave|omit|excluding|except|all but)\b", "omission_word"),
    ("entity_binding", r"\b(it|them|this|that|these|those|the one|the other|ones)\b", "binding_pronoun"),
]

COMPILED_RULES = [(label, re.compile(pattern, flags=re.IGNORECASE), name) for label, pattern, name in RULES]


def suggest(caption: str) -> dict[str, str]:
    text = caption.strip()
    matched: list[str] = []
    labels: list[str] = []
    for label, pattern, rule_name in COMPILED_RULES:
        if pattern.search(text):
            matched.append(rule_name)
            if label not in labels:
                labels.append(label)

    # The first label follows the rule table's conservative priority order;
    # multiple matches are explicitly marked low-confidence for human review.
    if not labels:
        primary = "none"
        confidence = "low"
    elif len(labels) == 1:
        primary = labels[0]
        confidence = "medium"
    else:
        primary = labels[0]
        confidence = "low"

    low = text.lower()
    if re.search(r"\b(remove|without|no longer|eliminate|delete|erase|leave only|keep only|all but|omit)\b", low):
        polarity = "remove"
    elif re.search(r"\b(replace|change|switch|swap|turn|transform|convert)\b", low):
        polarity = "replace"
    elif re.search(r"\b(move|put|place|position)\b", low):
        polarity = "move"
    elif re.search(r"\b(add|insert|include)\b", low):
        polarity = "add"
    elif re.search(r"\b(bigger|larger|more|darker|lighter|brighter|increase|intensif)\b", low):
        polarity = "intensify"
    elif re.search(r"\b(smaller|less|weaker|decrease|reduce)\b", low):
        polarity = "reduce"
    elif re.search(r"\b(similar|same|different|compare|matching)\b", low):
        polarity = "compare"
    else:
        polarity = "unclear"

    additional = [label for label in labels if label != primary]
    return {
        "suggested_primary_mechanism": primary,
        "suggested_additional_tags": "|".join(additional),
        "suggested_operation_polarity": polarity,
        "suggested_grounding": "unclear",
        "matched_rules": "|".join(matched),
        "confidence": confidence,
        "review_required": "yes",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = [
        "pairid",
        "annotation_group",
        "caption",
        "suggested_primary_mechanism",
        "suggested_additional_tags",
        "suggested_operation_polarity",
        "suggested_grounding",
        "matched_rules",
        "confidence",
        "review_required",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            suggestion = suggest(row.get("caption", ""))
            writer.writerow({
                "pairid": row.get("pairid", ""),
                "annotation_group": row.get("annotation_group", ""),
                "caption": row.get("caption", ""),
                **suggestion,
            })
    print(f"wrote={args.output} rows={len(rows)}")
    print("status=caption_only_preannotation; human_visual_review_required=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
