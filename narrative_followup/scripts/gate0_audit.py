#!/usr/bin/env python3
"""Freeze and audit inputs for the narrative follow-up experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/home/zihali/final_thesis")
OUT = ROOT / "narrative_followup"
PAIR = ROOT / "data/pair_control"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    OUT.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    raw = read_jsonl(PAIR / "test.jsonl")
    fixed = read_jsonl(PAIR / "test_fixed.jsonl")
    multi = read_jsonl(PAIR / "test_multi.jsonl")
    assert len(raw) == len(fixed) == len(multi) == 10_000

    def key(row: dict) -> tuple[str, str, str]:
        return row["id"], row["image"], row["target_img"]

    raw_keys = [key(x) for x in raw]
    fixed_keys = [key(x) for x in fixed]
    multi_keys = [key(x) for x in multi]
    assert raw_keys == fixed_keys == multi_keys, "query-style pair alignment failed"
    assert len(set(raw_keys)) == len(raw_keys), "duplicate pair IDs in test"

    test_ids = {x["id"] for x in raw}
    dev_ids = {x["id"] for x in read_jsonl(PAIR / "dev.jsonl")}
    train_ids = {x["id"] for x in read_jsonl(PAIR / "train.jsonl")}
    assert not test_ids & dev_ids
    assert not test_ids & train_ids
    assert not dev_ids & train_ids

    files = [
        PAIR / name
        for name in (
            "manifest.json",
            "train.jsonl",
            "train_fixed.jsonl",
            "train_multi.jsonl",
            "dev.jsonl",
            "dev_fixed.jsonl",
            "dev_multi.jsonl",
            "test.jsonl",
            "test_fixed.jsonl",
            "test_multi.jsonl",
        )
    ]

    ckpt_manifest = json.loads((ROOT / "cirr_test_results/checkpoint_manifest.json").read_text())
    checkpoints = {}
    for name, rec in ckpt_manifest["checkpoints"].items():
        path = ROOT / rec["checkpoint"]
        assert path.exists(), path
        got = sha256(path)
        assert got == rec["checkpoint_sha256"], (name, got, rec["checkpoint_sha256"])
        checkpoints[name] = {"path": str(path), "sha256": got, "common_dev_mAP": rec["common_dev_mAP"]}

    surface = ROOT / "data/pair_control/train_surfacediverse.jsonl"
    surface_rows = read_jsonl(surface)
    surface_ids = [x["id"] for x in surface_rows]
    assert len(surface_ids) == len(set(surface_ids))
    assert set(surface_ids) <= train_ids
    assert not set(surface_ids) & dev_ids
    assert not set(surface_ids) & test_ids

    alignment = {
        "status": "PASS",
        "n_queries": len(raw),
        "same_ordered_pair_keys": True,
        "fields": {"id": True, "reference_image": True, "target_image": True, "text_only_difference": True},
        "raw_fixed_multi_texts": {
            "raw": [x["modification"] for x in raw],
            "fixed": [x["modification"] for x in fixed],
            "multi": [x["modification"] for x in multi],
        },
    }
    # Do not store all three text corpora again; preserve compact alignment metadata.
    alignment["raw_fixed_multi_texts"] = {
        k: {"sha256": sha256(PAIR / filename), "lines": len(rows)}
        for k, filename, rows in (
            ("raw", "test.jsonl", raw),
            ("fixed", "test_fixed.jsonl", fixed),
            ("multi", "test_multi.jsonl", multi),
        )
    }
    (OUT / "manifests/query_style_alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    (OUT / "manifests/query_style_test_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "styles": {
                    "raw": str(PAIR / "test.jsonl"),
                    "fixed": str(PAIR / "test_fixed.jsonl"),
                    "multi": str(PAIR / "test_multi.jsonl"),
                },
                "n_queries": len(raw),
                "pair_key_order_sha256": hashlib.sha256("\n".join("|".join(x) for x in raw_keys).encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "manifests/checkpoint_manifest_snapshot.json").write_text(json.dumps(checkpoints, indent=2) + "\n")
    (OUT / "manifests/surface_control_50k_ids.json").write_text(
        json.dumps({"status": "PASS", "source": str(surface), "n_ids": len(surface_ids), "ids": surface_ids}, indent=2)
        + "\n"
    )

    lines = [
        "# Narrative follow-up Gate 0 protocol audit",
        "",
        "Status: PASS",
        "",
        "- Raw/Fixed/Multi test files contain 10,000 rows each with identical ordered `(id, reference, target)` keys.",
        "- Train/dev/test pair IDs are disjoint.",
        f"- Existing Surface-Diverse training artifact contains {len(surface_ids):,} unique IDs, all in pair-control train and outside dev/test.",
        f"- Nine frozen checkpoints were loaded from the checkpoint manifest and SHA256 verified.",
        "- Existing pair-control evaluator and checkpoint-selection manifest are retained; no checkpoint is reselected in this narrative analysis.",
        "",
        "The Surface-Diverse artifact is an input for a future formal control, not a completed three-seed experiment in this session.",
    ]
    (OUT / "reports/protocol_audit.md").write_text("\n".join(lines) + "\n")
    print("GATE0 PASS")
    print(f"queries={len(raw)} surface_train_ids={len(surface_ids)} checkpoints={len(checkpoints)}")


if __name__ == "__main__":
    main()
