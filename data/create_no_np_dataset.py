#!/usr/bin/env python3
"""Create a deterministic MerdCIR dataset without noun-phrase annotations.

The source JSONL may contain ``nps`` and the intermediate ``target_image`` /
``target_img`` fields produced by the rewrite pipeline.  The output is
canonicalized for :class:`data.dataset.MerdCIRDataset`, with no ``nps`` key.

The split is sampled by record index with a local ``random.Random`` instance,
so it is independent of Python's process-global random state and reproducible
from the source file plus ``--seed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_SEED = 114514


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield record


def canonicalize(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    out = dict(record)

    # The rewrite pipeline uses target_image; NP extraction emits target_img.
    # Prefer a real target_img, otherwise promote target_image.
    if not out.get("target_img") and out.get("target_image"):
        out["target_img"] = out["target_image"]
    out.pop("target_image", None)

    # Failed rewrite jobs can leave a record without either target key.  Such
    # a record cannot be consumed by MerdCIRDataset, so exclude it before the
    # seeded split is sampled.
    if not out.get("target_img"):
        return None

    # This is the defining property of the generated dataset.
    out.pop("nps", None)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/eval JSONL dataset without nps."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/merdcir_np/merged.jsonl"),
        help="Source JSONL containing MerdCIR records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/merdcir_no_np_seed114514"),
        help="Directory for train.jsonl, eval.jsonl, and manifest.json.",
    )
    parser.add_argument("--eval-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output JSONL/manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.eval_size <= 0:
        raise ValueError("--eval-size must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    manifest_path = output_dir / "manifest.json"
    outputs = (train_path, eval_path, manifest_path)
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace: " + ", ".join(existing)
            )

    total_source = sum(1 for _ in iter_records(args.input_jsonl))
    total = sum(
        1 for record in iter_records(args.input_jsonl) if canonicalize(record) is not None
    )
    if total <= args.eval_size:
        raise ValueError(f"Source has {total} records, not enough for eval-size={args.eval_size}")
    eval_indices = set(random.Random(args.seed).sample(range(total), args.eval_size))

    counts = {"train": 0, "eval": 0}
    with train_path.open("w", encoding="utf-8") as train_out, eval_path.open(
        "w", encoding="utf-8"
    ) as eval_out:
        valid_index = 0
        for record in iter_records(args.input_jsonl):
            normalized = canonicalize(record)
            if normalized is None:
                continue
            target = eval_out if valid_index in eval_indices else train_out
            split = "eval" if valid_index in eval_indices else "train"
            target.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1
            valid_index += 1

    manifest = {
        "dataset": "MerdCIR no-NP",
        "source_jsonl": str(args.input_jsonl),
        "source_sha256": sha256_file(args.input_jsonl),
        "seed": args.seed,
        "split_method": "random record-index sample for eval; source order preserved within each split",
        "source_records": total_source,
        "valid_records": total,
        "skipped_invalid_target_records": total_source - total,
        "records": counts,
        "np_field_present": False,
        "files": {
            "train.jsonl": {"records": counts["train"], "sha256": sha256_file(train_path)},
            "eval.jsonl": {"records": counts["eval"], "sha256": sha256_file(eval_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
