#!/usr/bin/env python3
"""Analyze human rating CSVs and generate visual PDF review packets."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "human_rating_results"
SAMPLES_PATH = ROOT / "samples.json"
ANNOTATORS = ("Annotator_1", "Annotator_2")
SPLIT_SIZE = 80
METRICS = (
    ("intent_preservation", "Intent Preservation"),
    ("naturalness", "Naturalness"),
    ("discriminativeness", "Discriminativeness"),
    ("harmful_omission_or_hallucination", "Harmful Omission / Hallucination"),
)
SYSTEMS = ("mtcir", "merdcir")


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def load_samples() -> list[dict[str, Any]]:
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotator in ANNOTATORS:
        path = RESULTS_DIR / f"evaluation_results_{annotator}.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_source_file"] = path.name
                rows.append(row)
    return rows


def latest_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("annotator", ""), row.get("sample_id", ""))].append(row)

    latest = []
    duplicate_counts = {}
    for key, values in grouped.items():
        if len(values) > 1:
            duplicate_counts[key] = len(values)
        latest.append(max(values, key=lambda row: parse_timestamp(row.get("timestamp", ""))))
    return latest, duplicate_counts


def assigned_samples(samples: list[dict[str, Any]], annotator: str) -> list[dict[str, Any]]:
    if annotator == "Annotator_1":
        return samples[:SPLIT_SIZE]
    if annotator == "Annotator_2":
        return samples[SPLIT_SIZE : SPLIT_SIZE * 2]
    raise ValueError(f"Unknown annotator: {annotator}")


def resolve_image(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def wrap_text(text: Any, width: int = 88, max_lines: int | None = None) -> str:
    wrapped = textwrap.wrap(str(text or ""), width=width)
    if max_lines is not None and len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1].rstrip(".") + " ..."
    return "\n".join(wrapped)


def image_or_placeholder(ax: Any, path: Path, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if path.is_file():
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
    else:
        ax.text(0.5, 0.5, f"Missing image:\n{path}", ha="center", va="center", fontsize=9)


def add_sample_page(pdf: PdfPages, sample: dict[str, Any], title: str, details: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    ax_ref = fig.add_axes((0.06, 0.58, 0.40, 0.30))
    ax_tgt = fig.add_axes((0.54, 0.58, 0.40, 0.30))
    image_or_placeholder(ax_ref, resolve_image(sample.get("reference_path")), "Reference")
    image_or_placeholder(ax_tgt, resolve_image(sample.get("target_path")), "Target")

    text_ax = fig.add_axes((0.06, 0.06, 0.88, 0.46))
    text_ax.axis("off")
    parts = [
        f"Sample ID: {sample.get('id')}",
        f"Reference path: {sample.get('reference_path')}",
        f"Target path: {sample.get('target_path')}",
        "",
        "MTCIR text:",
        wrap_text(sample.get("mtcir_text"), max_lines=5),
        "",
        "MiERDCIR text:",
        wrap_text(sample.get("merdcir_text"), max_lines=5),
    ]
    if details:
        parts.extend(["", *details])
    text_ax.text(0, 1, "\n".join(parts), ha="left", va="top", fontsize=9, family="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def generate_missing_pdf(missing: list[tuple[str, int, dict[str, Any]]]) -> Path | None:
    if not missing:
        return None
    out_path = RESULTS_DIR / "missing_annotation_entries.pdf"
    with PdfPages(out_path) as pdf:
        for annotator, local_index, sample in missing:
            add_sample_page(
                pdf,
                sample,
                f"Missing Annotation: {annotator} item {local_index + 1} / {SPLIT_SIZE}",
                ["Complete this entry manually in the annotation app or CSV."],
            )
    return out_path


def score_lines(row: dict[str, Any]) -> list[str]:
    lines = ["Scores:"]
    for key, label in METRICS:
        mtcir = row.get(f"mtcir_{key}", "")
        merdcir = row.get(f"merdcir_{key}", "")
        lines.append(f"- {label}: MTCIR={mtcir}, MiERDCIR={merdcir}")
    lines.extend(
        [
            f"Fail case: {row.get('is_mierdcir_fail_case')}",
            f"Annotator: {row.get('annotator')}",
            f"Timestamp: {row.get('timestamp')}",
        ]
    )
    if row.get("notes"):
        lines.extend(["Notes:", wrap_text(row.get("notes"), width=88, max_lines=4)])
    return lines


def generate_fail_case_pdf(rows: list[dict[str, Any]], sample_by_id: dict[str, dict[str, Any]]) -> Path:
    out_path = RESULTS_DIR / "annotated_mierdcir_fail_cases.pdf"
    fail_rows = [row for row in rows if parse_bool(row.get("is_mierdcir_fail_case"))]
    with PdfPages(out_path) as pdf:
        if not fail_rows:
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis("off")
            plt.text(0.5, 0.5, "No annotated MiERDCIR fail cases found.", ha="center", va="center")
            pdf.savefig(fig)
            plt.close(fig)
        for i, row in enumerate(fail_rows, start=1):
            sample = sample_by_id.get(row.get("sample_id", ""), row)
            title = f"MiERDCIR Fail Case {i} / {len(fail_rows)}"
            add_sample_page(pdf, sample, title, score_lines(row))
    return out_path


def summarize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        annotator = row.get("annotator", "")
        for system in SYSTEMS:
            for metric_key, _ in METRICS:
                value = parse_float(row.get(f"{system}_{metric_key}"))
                if value is not None:
                    groups[("All", system, metric_key)].append(value)
                    groups[(annotator, system, metric_key)].append(value)

    for annotator in ("All", *ANNOTATORS):
        for system in SYSTEMS:
            for metric_key, metric_label in METRICS:
                values = groups.get((annotator, system, metric_key), [])
                summary.append(
                    {
                        "annotator": annotator,
                        "system": system,
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "n": len(values),
                        "mean": mean(values) if values else "",
                        "std": stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else ""),
                    }
                )
    return summary


def write_summary_csv(summary: list[dict[str, Any]]) -> Path:
    out_path = RESULTS_DIR / "human_rating_score_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["annotator", "system", "metric", "metric_label", "n", "mean", "std"])
        writer.writeheader()
        for row in summary:
            formatted = dict(row)
            for key in ("mean", "std"):
                if isinstance(formatted[key], float):
                    formatted[key] = f"{formatted[key]:.4f}"
            writer.writerow(formatted)
    return out_path


def write_report(
    samples: list[dict[str, Any]],
    latest: list[dict[str, Any]],
    missing: list[tuple[str, int, dict[str, Any]]],
    duplicate_counts: dict[tuple[str, str], int],
    summary: list[dict[str, Any]],
    missing_pdf: Path | None,
    fail_pdf: Path,
    summary_csv: Path,
) -> Path:
    out_path = RESULTS_DIR / "human_rating_analysis_report.md"
    counts = Counter(row.get("annotator", "") for row in latest)
    fail_counts = Counter(row.get("annotator", "") for row in latest if parse_bool(row.get("is_mierdcir_fail_case")))

    by_key = {(row["annotator"], row["system"], row["metric"]): row for row in summary}
    lines = [
        "# Human Rating Analysis Report",
        "",
        "## Completeness",
        f"- Expected total annotations: {SPLIT_SIZE * 2}",
        f"- Latest unique annotations found: {len(latest)}",
    ]
    for annotator in ANNOTATORS:
        lines.append(f"- {annotator}: {counts.get(annotator, 0)} / {SPLIT_SIZE}")
    lines.append(f"- Missing annotations: {len(missing)}")
    lines.append(f"- Duplicate `(annotator, sample_id)` entries: {len(duplicate_counts)}")
    if missing_pdf:
        lines.append(f"- Missing-entry PDF: `{missing_pdf.name}`")
    else:
        lines.append("- Missing-entry PDF: not created because no entries are missing")

    lines.extend(["", "## Score Summary", ""])
    for annotator in ("All", *ANNOTATORS):
        lines.extend([f"### {annotator}", "", "| System | Metric | n | Mean | Std |", "|---|---|---:|---:|---:|"])
        for system in SYSTEMS:
            for metric_key, metric_label in METRICS:
                row = by_key[(annotator, system, metric_key)]
                mean_value = row["mean"]
                std_value = row["std"]
                mean_text = f"{mean_value:.2f}" if isinstance(mean_value, float) and not math.isnan(mean_value) else ""
                std_text = f"{std_value:.2f}" if isinstance(std_value, float) and not math.isnan(std_value) else ""
                lines.append(f"| {system} | {metric_label} | {row['n']} | {mean_text} | {std_text} |")
        lines.append("")

    lines.extend(["## Annotator Difference", ""])
    lines.append("Annotators rated disjoint sample splits, so these differences mix annotator behavior with sample-split difficulty.")
    lines.extend(["", "| System | Metric | Annotator_1 Mean | Annotator_2 Mean | A2 - A1 |", "|---|---|---:|---:|---:|"])
    for system in SYSTEMS:
        for metric_key, metric_label in METRICS:
            a1 = by_key[("Annotator_1", system, metric_key)]["mean"]
            a2 = by_key[("Annotator_2", system, metric_key)]["mean"]
            diff = a2 - a1 if isinstance(a1, float) and isinstance(a2, float) else ""
            lines.append(f"| {system} | {metric_label} | {a1:.2f} | {a2:.2f} | {diff:.2f} |")

    lines.extend(
        [
            "",
            "## Fail Cases",
            f"- Annotated MiERDCIR fail cases: {sum(fail_counts.values())}",
            f"- Annotator_1 fail cases: {fail_counts.get('Annotator_1', 0)}",
            f"- Annotator_2 fail cases: {fail_counts.get('Annotator_2', 0)}",
            f"- Fail-case PDF: `{fail_pdf.name}`",
            f"- Machine-readable score summary: `{summary_csv.name}`",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    samples = load_samples()
    rows = load_rows()
    latest, duplicate_counts = latest_rows(rows)
    sample_by_id = {str(sample["id"]): sample for sample in samples}
    latest_keys = {(row.get("annotator", ""), row.get("sample_id", "")) for row in latest}

    missing = []
    for annotator in ANNOTATORS:
        for local_index, sample in enumerate(assigned_samples(samples, annotator)):
            key = (annotator, str(sample["id"]))
            if key not in latest_keys:
                missing.append((annotator, local_index, sample))

    missing_pdf = generate_missing_pdf(missing)
    fail_pdf = generate_fail_case_pdf(latest, sample_by_id)
    summary = summarize_scores(latest)
    summary_csv = write_summary_csv(summary)
    report = write_report(samples, latest, missing, duplicate_counts, summary, missing_pdf, fail_pdf, summary_csv)

    print(f"Rows loaded: {len(rows)}")
    print(f"Latest unique rows: {len(latest)}")
    print(f"Missing annotations: {len(missing)}")
    print(f"Duplicate keys: {len(duplicate_counts)}")
    print(f"Fail cases: {sum(1 for row in latest if parse_bool(row.get('is_mierdcir_fail_case')))}")
    print(f"Report: {report}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Missing PDF: {missing_pdf or 'not created'}")
    print(f"Fail-case PDF: {fail_pdf}")


if __name__ == "__main__":
    main()
