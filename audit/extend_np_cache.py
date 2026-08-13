#!/usr/bin/env python3
"""Extend the NP cache to cover mid/late train-split regions so the
matched-gradient probe can sample early/mid/late stages (plan §8.8)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "reproduction_evidence" / "np_cache"
sys.path.insert(0, str(SRC))

from Parser import Parser  # noqa: E402

MAX_NPS = 3
CACHE = OUT / f"cirr_train_maxnps{MAX_NPS}_noun_phrases.json"
CAPTIONS = json.loads((ROOT / "CIRR" / "cap.rc2.train.json").read_text())
CAPTION_TEXTS = [row["caption"] for row in CAPTIONS]
N = len(CAPTION_TEXTS)

def serialize(np_info):
    return {
        "nps": np_info.get("nps", []),
        "spans": [
            {"left": int(span.left), "right": int(span.right)}
            for span in np_info.get("spans", [])
        ],
    }

def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"existing cache: {len(cache)} entries, dataset size: {N}")

    parser = Parser(None, None)  # matches scan usage (Parser(text_model, tokenizer) may be None)
    # regions: early(0..64 already cached), mid, late
    regions = {
        "mid": list(range(N // 2, N // 2 + 64)),
        "late": list(range(N - 64, N)),
    }
    for name, idxs in regions.items():
        added = 0
        for i in idxs:
            caption = CAPTION_TEXTS[i]
            if caption in cache:
                continue
            try:
                np_info = parser.extract_noun_phrases(caption, max_nps=MAX_NPS)
            except Exception as e:
                print(f"  parser error at {i}: {e}")
                continue
            cache[caption] = serialize(np_info)
            added += 1
        print(f"{name}: added {added} entries")

    CACHE.write_text(json.dumps(cache))
    print(f"cache now: {len(cache)} entries -> {CACHE}")


if __name__ == "__main__":
    main()
