#!/usr/bin/env python3
"""Build a strict surface-only control from the frozen Fixed corpus.

The semantic core is copied byte-for-byte from Fixed; only a deterministic
sentence frame is added. This avoids the content and length confounds in the
older VLM-generated 50K Surface-Diverse exploratory corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FRAMES = (
    "Show me a photo like this one, with {core}",
    "I am looking for a similar image featuring {core}",
    "Find a version of this scene that has {core}",
    "I want to see this scene with {core}",
    "Retrieve an image showing {core}",
    "A version of this image featuring {core}",
)


def frame_index(row_id: str) -> int:
    return int(hashlib.sha256(str(row_id).encode()).hexdigest()[:8], 16) % len(FRAMES)


def convert(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    frame_counts = [0] * len(FRAMES)
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            core = str(row.get("modification", "")).strip()
            if not core:
                raise ValueError(f"empty Fixed modification at {input_path}:{line_no}")
            idx = frame_index(row_id)
            out = dict(row)
            out["modification"] = FRAMES[idx].format(core=core.rstrip(". ")) + "."
            out["surface_frame"] = idx + 1
            out.pop("nps", None)
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
            frame_counts[idx] += 1
    return {"lines": count, "frame_counts": frame_counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("data/pair_control"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/pair_control_surface_v2"))
    args = ap.parse_args()
    summary = {"source": {}, "outputs": {}}
    for split in ("train", "dev", "test"):
        src = args.input_dir / f"{split}_fixed.jsonl"
        dst = args.output_dir / f"{split}.jsonl"
        summary["source"][src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
        summary["outputs"][dst.name] = convert(src, dst)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "protocol": "surface-only control: Fixed semantic core plus one of six deterministic frames",
                "frames": list(FRAMES),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
