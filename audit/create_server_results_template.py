#!/usr/bin/env python3
"""Create the blank official-CIRR result sheet from the frozen manifest."""
import csv
import json
from pathlib import Path


def main() -> None:
    manifest = json.loads(Path("cirr_test_results/submission_manifest.json").read_text())
    rows = []
    for item in manifest["submissions"]:
        condition, seed = item["condition_seed"].rsplit("_s", 1)
        rows.append({
            "submission_date": "TBD",
            "condition": condition,
            "seed": seed,
            "metric": item["metric"],
            "file_sha256": item["sha256"],
            "server_R1": "",
            "server_R5": "",
            "server_R10": "",
            "server_R50": "",
            "server_subset_R1": "",
            "server_subset_R2": "",
            "server_subset_R3": "",
            "notes": "",
        })
    out = Path("cirr_test_results/server_results.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["condition"], int(r["seed"]), r["metric"])))
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
