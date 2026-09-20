#!/usr/bin/env python3
"""Build a provenance manifest for locally validated D3-full CIRR exports.

This intentionally does not contact the CIRR server.  It records the frozen
checkpoint, prediction and validator hashes so a later upload can be added
without changing the local export set.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cirr_test_results" / "d3_full_lasco_no_cc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for seed, val_loss in ((0, "2.192"), (1, "2.185"), (2, "2.172")):
        checkpoint = (
            ROOT
            / "reproduction_experiments/ConText-CIR-09052026/reproduction_evidence"
            / f"session_d3_full_lasco_no_cc_s{seed}"
            / f"epoch=0-step=1623-val_loss={val_loss}e+00.ckpt"
        )
        stem = f"d3_full_lasco_no_cc_s{seed}_0"
        for metric, prefix in (("recall", ""), ("recall_subset", "subset_")):
            prediction = OUT / f"{prefix}{stem}.json"
            validation = Path(str(prediction) + ".validation.json")
            if not checkpoint.exists() or not prediction.exists() or not validation.exists():
                raise SystemExit(f"missing package input for seed={seed}, metric={metric}")
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            if not validation_payload.get("all_checks_passed"):
                raise SystemExit(f"validator did not pass: {validation}")
            rows.append(
                {
                    "condition_seed": f"D3-full_s{seed}",
                    "condition": "D3-full LaSCo no-Text-CC",
                    "seed": seed,
                    "metric": metric,
                    "file": str(prediction.relative_to(ROOT)),
                    "sha256": sha256(prediction),
                    "bytes": prediction.stat().st_size,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "validation_manifest": str(validation.relative_to(ROOT)),
                    "validator": "PASS",
                    "server_status": "not_submitted",
                    "server_results": {},
                }
            )

    payload = {
        "protocol": "local D3-full LaSCo no-Text-CC CIRR test1 export",
        "generated": str(date.today()),
        "checkpoint_selection": "frozen from existing CIRR validation-transfer evaluations; test results not used",
        "server_submission": "not performed",
        "submissions": rows,
    }
    (OUT / "submission_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "submission_date", "condition", "seed", "metric", "file_sha256",
        "server_R1", "server_R5", "server_R10", "server_R50",
        "server_subset_R1", "server_subset_R2", "server_subset_R3", "notes",
    ]
    with (OUT / "server_results_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "submission_date": "TBD",
                    "condition": row["condition"],
                    "seed": row["seed"],
                    "metric": row["metric"],
                    "file_sha256": row["sha256"],
                    "notes": "Upload separately; leave blank until server response is recorded.",
                }
            )
    print(f"wrote {OUT / 'submission_manifest.json'} and {OUT / 'server_results_template.csv'}")


if __name__ == "__main__":
    main()
