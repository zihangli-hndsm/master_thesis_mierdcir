#!/usr/bin/env python3
"""Summarize available intent labels and text-surface coverage.

This deliberately separates the labeled exploratory VLM corpus from the
deterministic Surface-v2 control; it does not invent intent labels for the
latter.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

from transformers import CLIPTokenizerFast


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/pair_control/train_surfacediverse_raw.jsonl"
IDS = ROOT / "narrative_followup/manifests/surface_control_50k_ids.json"
CONTROL = ROOT / "data/pair_control_surface_50k"
OUT_JSON = ROOT / "narrative_followup/metrics/intent_category_metrics.json"
OUT_MD = ROOT / "narrative_followup/reports/intent_coverage_report.md"


def js_divergence(counts: dict[str, int]) -> float:
    keys = list(counts)
    total = sum(counts.values())
    p = [counts[k] / total for k in keys]
    q = [1 / len(keys)] * len(keys)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl_pm = sum(a * math.log(a / b, 2) for a, b in zip(p, m) if a)
    kl_qm = sum(a * math.log(a / b, 2) for a, b in zip(q, m) if a)
    return (kl_pm + kl_qm) / 2


def text_stats(path: Path, tokenizer: CLIPTokenizerFast) -> dict[str, float]:
    chars, toks, nps = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("modification", "")
            chars.append(len(text))
            toks.append(len(tokenizer(text)["input_ids"]))
            nps.append(len(row.get("nps", [])))
    def avg(values):
        return sum(values) / len(values)
    return {"rows": len(chars), "mean_chars": avg(chars), "mean_clip_tokens": avg(toks), "mean_nps": avg(nps)}


def main() -> None:
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    counts: collections.Counter[str] = collections.Counter()
    raw_ids: set[str] = set()
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw_ids.add(str(row["id"]))
            counts[str(row.get("merdcir_intent_scenario", "unknown"))] += 1
    frozen = json.loads(IDS.read_text(encoding="utf-8"))
    frozen_ids = set(map(str, frozen["ids"]))
    frame_counts: collections.Counter[str] = collections.Counter()
    with (CONTROL / "train_surface.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            frame_counts[str(row.get("surface_frame", "unknown"))] += 1
    payload = {
        "labeled_exploratory_vlm_corpus": {
            "rows": sum(counts.values()),
            "scenario_counts": dict(sorted(counts.items())),
            "js_divergence_to_uniform_bits": js_divergence(dict(counts)),
            "frozen_50k_ids_covered": len(raw_ids & frozen_ids),
        },
        "deterministic_surface_control": {
            "rows": sum(frame_counts.values()),
            "frame_counts": dict(sorted(frame_counts.items())),
            "note": "surface frames are controlled lexical templates, not semantic intent labels",
        },
        "matched_50k_text_stats": {
            name: text_stats(CONTROL / f"train_{name}.jsonl", tokenizer)
            for name in ("fixed", "surface", "multi")
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Intent coverage and text-surface audit",
        "",
        "This report distinguishes the labeled exploratory VLM corpus from the deterministic Surface-v2 control. The latter is intentionally not assigned semantic intent labels.",
        "",
        "## Exploratory VLM corpus",
        "",
        f"- Rows: **{payload['labeled_exploratory_vlm_corpus']['rows']:,}**",
        f"- Frozen matched IDs covered: **{payload['labeled_exploratory_vlm_corpus']['frozen_50k_ids_covered']:,}**",
        f"- JS divergence from uniform scenario coverage: **{payload['labeled_exploratory_vlm_corpus']['js_divergence_to_uniform_bits']:.6f} bits**",
        "",
        "| Scenario | Count | Proportion |",
        "|---|---:|---:|",
    ]
    total = sum(counts.values())
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value:,} | {value / total:.4f} |")
    lines += ["", "## Deterministic Surface-v2 frame balance", "", "| Frame | Count |", "|---|---:|"]
    for key, value in sorted(frame_counts.items()):
        lines.append(f"| {key} | {value:,} |")
    lines += ["", "## Matched 50K text statistics", "", "| Condition | Rows | Mean chars | Mean CLIP tokens | Mean NPs |", "|---|---:|---:|---:|---:|"]
    for name, stats in payload["matched_50k_text_stats"].items():
        lines.append(f"| {name} | {stats['rows']:,} | {stats['mean_chars']:.2f} | {stats['mean_clip_tokens']:.2f} | {stats['mean_nps']:.2f} |")
    lines += ["", "Interpretation: the exploratory corpus has near-balanced labeled scenario coverage, while the formal Surface-50K control isolates deterministic surface framing around the Fixed semantic core. A semantic intent claim still requires a common rubric and manual validation across benchmark query styles.", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
