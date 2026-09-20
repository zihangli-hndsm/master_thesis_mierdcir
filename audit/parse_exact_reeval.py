#!/usr/bin/env python3
"""Parse run_exact_reeval.sh output into a compact machine-readable table."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("checkpoints/pair_control_exact_metrics_20260816.json"))
    args = ap.parse_args()
    log = args.log or Path(sorted(glob.glob("logs/exact_reeval_*.log"))[-1])
    text = log.read_text(encoding="utf-8")
    records = {}
    condition = None
    dataset = None
    pattern = re.compile(r"(Recall@1|Recall@5|Recall@10|Recall@50|mAP): ([0-9.]+)")
    for line in text.splitlines():
        m = re.match(r"=== (\w+_s\d+) (MTCIR|MerdCIR) \(exact\) ===", line)
        if m:
            condition, dataset = m.group(1), m.group(2)
            records.setdefault(condition, {}).setdefault(dataset, {})
            continue
        if condition and dataset:
            metric = pattern.match(line)
            if metric:
                records[condition][dataset][metric.group(1)] = float(metric.group(2))
    expected = {f"{cond}_s{seed}" for cond in ("raw", "fixed", "multi") for seed in ("42", "123", "2025")}
    missing = sorted(expected - records.keys())
    if missing:
        raise SystemExit(f"missing conditions in {log}: {missing}")
    payload = {"log": str(log), "metrics": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
