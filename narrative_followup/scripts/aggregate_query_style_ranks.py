#!/usr/bin/env python3
"""Validate and aggregate the 27 exact-gallery query-rank exports."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "narrative_followup/metrics"
TEST = ROOT / "data/pair_control/test.jsonl"
OUT = METRICS / "query_style_ranks.csv"
PARQUET = METRICS / "query_style_ranks.parquet"
MANIFEST = METRICS / "query_style_rank_manifest.json"
REPORT = ROOT / "narrative_followup/reports/query_style_rank_audit.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_ids() -> list[str]:
    ids = []
    with TEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(str(row.get("id", row.get("pairid"))))
    return ids


def parse_name(path: Path) -> tuple[str, int, str]:
    # query_ranks_<condition>_s<seed>_<style>.csv
    stem = path.stem.removeprefix("query_ranks_")
    condition, seed, style = stem.rsplit("_", 2)
    return condition, int(seed.removeprefix("s")), style


def main() -> None:
    ids = expected_ids()
    if len(ids) != 10_000 or len(set(ids)) != len(ids):
        raise SystemExit(f"unexpected pair-control test IDs: {len(ids)} rows, {len(set(ids))} unique")
    expected = {(condition, seed, style) for condition in ("raw", "fixed", "multi") for seed in (42, 123, 2025) for style in ("raw", "fixed", "multi")}
    files = sorted(METRICS.glob("query_ranks_*.csv"))
    seen = set()
    all_rows = []
    cells = []
    for path in files:
        condition, seed, style = parse_name(path)
        key = (condition, seed, style)
        if key in seen:
            raise SystemExit(f"duplicate cell: {key}")
        seen.add(key)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(ids):
            raise SystemExit(f"{path}: expected {len(ids)} rows, found {len(rows)}")
        row_ids = [row["pair_id"] for row in rows]
        if row_ids != ids:
            raise SystemExit(f"{path}: pair_id order/set differs from pair-control test")
        ranks = []
        for row in rows:
            rank = int(row["target_rank"])
            if rank < 1:
                raise SystemExit(f"{path}: invalid rank {rank}")
            ranks.append(rank)
            all_rows.append({
                "condition": condition,
                "seed": seed,
                "query_style": style,
                "pair_id": row["pair_id"],
                "target_id": row["target_id"],
                "target_rank": rank,
            })
        recomputed = {
            "Recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
            "Recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
            "Recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
            "Recall@50": sum(rank <= 50 for rank in ranks) / len(ranks),
            "mAP": sum(1.0 / rank for rank in ranks) / len(ranks),
        }
        cells.append({
            "condition": condition,
            "seed": seed,
            "query_style": style,
            "rows": len(rows),
            "file": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            **recomputed,
        })
    if seen != expected:
        raise SystemExit(f"expected 27 cells, found {len(seen)}; missing={sorted(expected-seen)} extra={sorted(seen-expected)}")

    matrix = {}
    with (METRICS / "query_style_matrix.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            matrix[(row["condition"], int(row["seed"]), row["query_style"])] = row
    for cell in cells:
        expected_row = matrix[(cell["condition"], cell["seed"], cell["query_style"])]
        diffs = {metric: abs(cell[metric] - float(expected_row[metric])) for metric in ("Recall@1", "Recall@5", "Recall@10", "Recall@50", "mAP")}
        cell["max_abs_diff"] = max(diffs.values())
        # The original per-cell JSON stores metrics at six decimal places;
        # allow only the corresponding serialization error, not a substantive
        # metric discrepancy.
        cell["matrix_match"] = cell["max_abs_diff"] <= 1e-6
        if not cell["matrix_match"]:
            raise SystemExit(f"{cell['file']}: rank recomputation differs from matrix: {diffs}")

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        fields = ["condition", "seed", "query_style", "pair_id", "target_id", "target_rank"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "status": "PASS",
        "expected_cells": 27,
        "rows_per_cell": len(ids),
        "aggregated_rows": len(all_rows),
        "pair_control_test_sha256": sha256(TEST),
        "max_matrix_abs_diff": max(cell["max_abs_diff"] for cell in cells),
        "cells": cells,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Query-style rank audit", "",
        "The 27 exact-gallery rank exports contain 10,000 pair-aligned rows per cell. "
        "Recall and mAP recomputed from the target ranks match `query_style_matrix.csv` "
        "within an absolute tolerance of 1e-6 (the source JSON precision).", "",
        f"- Cells: **{len(cells)}**", f"- Aggregated rows: **{len(all_rows):,}**",
        f"- Maximum matrix absolute difference: **{payload['max_matrix_abs_diff']:.3g}**", "",
        "The consolidated CSV and Parquet files are audit artifacts, not causal annotations. Query-style difficulty and surface-form differences remain part of the interpretation boundary.", "",
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": len(cells), "rows": len(all_rows), "max_matrix_abs_diff": payload["max_matrix_abs_diff"]}, indent=2))


if __name__ == "__main__":
    main()
