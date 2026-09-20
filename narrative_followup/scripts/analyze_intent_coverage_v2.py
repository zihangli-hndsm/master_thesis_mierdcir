#!/usr/bin/env python3
"""Descriptive, rule-based intent coverage scan for the final supplement.

This is deliberately an audit aid, not a semantic ground-truth annotator. Every
automatic label includes the matched rules and a confidence flag; a fixed 100
rows/source review sheet is emitted for manual validation.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "narrative_followup"

PATTERNS = {
    "negative_constraint": r"\b(no|not|without|avoid|exclude|remove|removed|don't|do not|doesn't|never|none)\b",
    "spatial_positional": r"\b(left|right|center|middle|behind|front|foreground|background|above|below|beside|next to|near|move|position|angle|viewpoint|close[- ]up|farther|closer)\b",
    "comparative_intensity": r"\b(more|less|darker|lighter|brighter|longer|shorter|larger|smaller|bigger|higher|lower|wider|narrower|deeper|stronger|slightly|much|increase|decrease)\b",
    "functional_affordance": r"\b(suitable|suited|for work|for sport|for formal|for a wedding|for an occasion|use|usable|function|purpose|professional|gala|decor|hiking|running)\b",
    "global_view": r"\b(background|scene|setting|atmosphere|vibe|lighting|light|dark|nighttime|daytime|outdoor|indoor|weather|blurred|color photo|black and white|overall)\b",
    "instance_level": r".*",
}
COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
ORDER = ["negative_constraint", "spatial_positional", "comparative_intensity", "functional_affordance", "global_view", "instance_level"]


def classify(text: str) -> tuple[str, str, str]:
    matched = [name for name in ORDER[:-1] if COMPILED[name].search(text or "")]
    primary = matched[0] if matched else "instance_level"
    confidence = "high" if len(matched) == 1 else ("medium" if matched else "low")
    return primary, "|".join(matched), confidence


def emit(source: str, rows: list[dict], out_rows: list[dict]) -> None:
    for row in rows:
        text = row["text"]
        label, matched, confidence = classify(text)
        out_rows.append({
            "source": source,
            "id": str(row["id"]),
            "text": text,
            "auto_primary_intent": label,
            "matched_rules": matched,
            "confidence": confidence,
        })


def read_jsonl(path: Path, text_keys: tuple[str, ...], id_key: str = "id") -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = next((row.get(k) for k in text_keys if row.get(k)), "")
            if isinstance(text, list):
                text = " ".join(map(str, text))
            out.append({"id": row.get(id_key, len(out)), "text": str(text)})
    return out


def main() -> None:
    sources: dict[str, list[dict]] = {}
    sources["MTCIR_test_raw"] = read_jsonl(ROOT / "data/pair_control/test.jsonl", ("modification",))
    sources["MerdCIR_test"] = read_jsonl(ROOT / "data/merdcir_np/eval/full_eval_subset.jsonl", ("modification", "merdcir_modification"))
    cirr = json.loads((ROOT / "data/CIRR/cap.rc2.val.json").read_text(encoding="utf-8"))
    sources["CIRR_val"] = [{"id": r.get("pairid", i), "text": str(r.get("caption", ""))} for i, r in enumerate(cirr)]
    fashion = []
    for path in sorted((ROOT / "data/FashionIQ/captions").glob("cap.*.val.json")):
        for i, r in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            captions = r.get("captions", [])
            fashion.append({"id": f"{path.stem}:{i}", "text": " ".join(map(str, captions))})
    sources["FashionIQ_val"] = fashion

    out_rows: list[dict] = []
    for source, rows in sources.items():
        emit(source, rows, out_rows)

    metrics = []
    for source in sources:
        subset = [r for r in out_rows if r["source"] == source]
        counts = Counter(r["auto_primary_intent"] for r in subset)
        total = len(subset)
        for label in ORDER:
            metrics.append({
                "source": source,
                "intent": label,
                "count": counts[label],
                "proportion": counts[label] / total if total else 0.0,
                "automatic_only": True,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics/intent_coverage_auto_v2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (OUT / "metrics/intent_query_labels_auto_v2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    # Fixed deterministic review set: 100 rows per source, stratified by the
    # automatic label when possible.
    review = []
    for source, rows in sources.items():
        subset = [r for r in out_rows if r["source"] == source]
        by_label = {label: [r for r in subset if r["auto_primary_intent"] == label] for label in ORDER}
        chosen = []
        for label in ORDER:
            take = max(1, min(100, len(subset)) // len(ORDER))
            chosen.extend(by_label[label][:take])
        chosen_ids = {r["id"] for r in chosen}
        chosen.extend(r for r in subset if r["id"] not in chosen_ids)
        review.extend(chosen[: min(100, len(subset))])
    for row in review:
        row["manual_intent"] = ""
        row["manual_valid"] = ""
        row["manual_notes"] = ""
    with (OUT / "annotations/intent_coverage_manual_review_v2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)

    report = [
        "# Intent coverage v2: automatic descriptive scan",
        "",
        "This is a rule-based descriptive scan. It is not ground truth and must not be used for causal claims before manual validation.",
        "",
        "| Source | Rows | Automatic labels | Manual review target |",
        "|---|---:|---|---:|",
    ]
    for source, rows in sources.items():
        counts = Counter(r["auto_primary_intent"] for r in out_rows if r["source"] == source)
        report.append(f"| {source} | {len(rows):,} | {dict(counts)} | {min(100, len(rows))} |")
    report += [
        "",
        "The emitted manual sheet contains blank `manual_intent`, `manual_valid`, and `manual_notes` columns. Until those fields are filled and checked, the automatic distributions are descriptive evidence only.",
        "",
    ]
    (OUT / "reports/intent_coverage_v2.md").write_text("\n".join(report), encoding="utf-8")
    print("sources", {k: len(v) for k, v in sources.items()})
    print("rows", len(out_rows), "review", len(review))


if __name__ == "__main__":
    main()
