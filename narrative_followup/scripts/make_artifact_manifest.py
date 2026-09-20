#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/zihali/final_thesis")
OUT = ROOT / "narrative_followup"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def main() -> None:
    paths = [
        ROOT / "data/pair_control/test.jsonl",
        ROOT / "data/pair_control/test_fixed.jsonl",
        ROOT / "data/pair_control/test_multi.jsonl",
        ROOT / "data/pair_control/train.jsonl",
        ROOT / "data/pair_control/dev.jsonl",
        ROOT / "data/pair_control/train_surfacediverse.jsonl",
        ROOT / "cirr_test_results/checkpoint_manifest.json",
    ]
    paths += sorted((ROOT / "checkpoints").glob("topk_pair_*/topk_epoch_0002_step_000600_*.pth.tar"))
    paths += sorted((OUT / "metrics").glob("query_*.json"))
    paths += sorted((OUT / "metrics").glob("*.csv"))
    paths += sorted((OUT / "metrics").glob("*.parquet"))
    paths += sorted((OUT / "predictions").rglob("*.json"))
    paths += sorted((OUT / "annotations").glob("*.csv"))
    paths += sorted((OUT / "annotations/contact_sheets").glob("*.pdf"))
    paths += [
        OUT / "annotations/ANNOTATION_PROTOCOL.md",
        OUT / "annotations/annotation_agreement_v2.md",
    ]
    files = []
    for p in paths:
        if p.is_file():
            files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "files": files,
        "commands": [
            "scripts/gate0_audit.py",
            "narrative_followup/scripts/run_query_style_matrix.sh",
            "narrative_followup/scripts/analyze_query_style_matrix.py",
            "narrative_followup/scripts/run_cirr_val_rank_exports.sh",
            "narrative_followup/scripts/analyze_cirr_rank_deltas.py",
        ],
    }
    (OUT / "artifact_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"manifest files={len(files)}")


if __name__ == "__main__":
    main()
