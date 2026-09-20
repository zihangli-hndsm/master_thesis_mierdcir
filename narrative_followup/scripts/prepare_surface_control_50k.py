#!/usr/bin/env python3
"""Prepare the matched 50K surface-control experiment inputs.

The existing ``pair_control_surface_v2/train_np.jsonl`` is only a partial NP
cache.  This script filters the complete deterministic surface-v2 corpus to
the frozen 50K IDs and reconstructs NP spans in the framed sentence using the
same CLIP tokenizer convention as ``scripts/train.py`` (1-based positions
after the BOS token).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import CLIPTokenizerFast


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def surface_nps(
    surface_text: str,
    fixed_text: str,
    fixed_nps: list[list[Any]],
    tokenizer: CLIPTokenizerFast,
) -> list[list[Any]]:
    """Map Fixed NP strings into the framed Surface sentence."""
    core_start = surface_text.find(fixed_text.rstrip(". "))
    if core_start < 0:
        raise ValueError(f"Fixed core not found in Surface text: {surface_text!r}")

    encoded = tokenizer(surface_text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded["offset_mapping"]
    fixed_encoded = tokenizer(fixed_text, return_offsets_mapping=True, add_special_tokens=True)
    fixed_offsets = fixed_encoded["offset_mapping"]
    output: list[list[Any]] = []
    for np_text, old_span in fixed_nps:
        np_text = str(np_text)
        old_start, old_end = map(int, old_span)
        if old_start < 1 or old_end >= len(fixed_offsets):
            raise ValueError(f"Invalid Fixed NP span {old_span!r} for {fixed_text!r}")
        fixed_char_start = fixed_offsets[old_start][0]
        fixed_char_end = fixed_offsets[old_end][1]
        if fixed_char_start >= fixed_char_end:
            raise ValueError(f"Empty Fixed NP span {old_span!r} for {fixed_text!r}")
        # The deterministic frame preserves the Fixed core byte-for-byte up
        # to harmless trailing punctuation/space normalization, so token
        # offsets transfer directly into the framed sentence.
        char_start = core_start + fixed_char_start
        char_end = core_start + fixed_char_end
        token_indices = [
            i for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_start >= char_start and tok_end <= char_end and tok_start < tok_end
        ]
        if not token_indices:
            raise ValueError(f"NP {np_text!r} did not map to CLIP tokens in {surface_text!r}")
        # gen_np.py records tokenizer indices directly: BOS is index 0, so the
        # training code later converts these 1-based positions with start-1.
        start, end = token_indices[0], token_indices[-1]
        if start < 1 or end > 77:
            raise ValueError(f"NP span out of CLIP range: {np_text!r} -> {(start, end)}")
        output.append([np_text, [start, end]])
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--ids", type=Path, default=Path("narrative_followup/manifests/surface_control_50k_ids.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/pair_control_surface_50k"))
    args = ap.parse_args()

    root = args.root.resolve()
    ids_payload = json.loads((root / args.ids).read_text(encoding="utf-8"))
    ids = [str(x) for x in ids_payload["ids"]]
    id_set = set(ids)
    if len(ids) != 50_000 or len(id_set) != len(ids):
        raise ValueError(f"Expected 50,000 unique frozen IDs, got {len(ids)} / {len(id_set)}")

    paths = {
        "fixed": root / "data/pair_control/train_fixed.jsonl",
        "multi": root / "data/pair_control/train_multi.jsonl",
        "surface": root / "data/pair_control_surface_v2/train.jsonl",
    }
    rows = {name: load_jsonl(path) for name, path in paths.items()}
    by_id = {
        name: {str(row["id"]): row for row in values}
        for name, values in rows.items()
    }
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    selected: dict[str, list[dict[str, Any]]] = {"fixed": [], "multi": [], "surface": []}
    for row_id in ids:
        missing = [name for name, mapping in by_id.items() if row_id not in mapping]
        if missing:
            raise ValueError(f"ID {row_id} missing from {missing}")
        fixed = by_id["fixed"][row_id]
        multi = by_id["multi"][row_id]
        surface = dict(by_id["surface"][row_id])
        if (fixed["image"], fixed["target_img"]) != (multi["image"], multi["target_img"]):
            raise ValueError(f"Pair mismatch for {row_id} between Fixed and Multi")
        if (fixed["image"], fixed["target_img"]) != (surface["image"], surface["target_img"]):
            raise ValueError(f"Pair mismatch for {row_id} between Fixed and Surface")
        surface["nps"] = surface_nps(
            surface["modification"], fixed["modification"], fixed.get("nps", []), tokenizer
        )
        surface["surface_control"] = "fixed_core_plus_deterministic_frame"
        selected["fixed"].append(fixed)
        selected["multi"].append(multi)
        selected["surface"].append(surface)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol": "matched 50K train subset; Fixed vs Multi vs deterministic Surface-v2",
        "source_ids": str(args.ids),
        "n_ids": len(ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "sources": {name: {"path": str(path), "sha256": sha256(path), "rows": len(rows[name])}
                     for name, path in paths.items()},
        "outputs": {},
    }
    for name, values in selected.items():
        out = args.output_dir / f"train_{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["outputs"][name] = {
            "path": str(out),
            "rows": len(values),
            "sha256": sha256(out),
            "first_id": values[0]["id"],
            "last_id": values[-1]["id"],
        }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
