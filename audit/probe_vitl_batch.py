#!/usr/bin/env python3
"""Probe the max stable per-forward batch size for ScheiCIR(ViT-L/14) on this GPU.

Random image/text tensors (no data pipeline); reports peak VRAM per batch and
the largest batch that completes a full forward+backward without OOM.
This batch will be used for the §8.4 batch-matched ViT-B re-runs too.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.full_model import ScheiCIR

TEXTS = [
    "make the background darker and add more people",
    "change the color of the dress to red",
    "remove the chair and add a table",
    "make the image brighter and more colorful",
    "add a logo of the bank and include some text",
    "replace the car with a bicycle",
    "make the sky blue and the grass greener",
    "add a dog next to the woman",
]


def make_batch(bs, device, tokenizer):
    imgs = torch.randn(bs, 3, 224, 224, device=device)
    texts = [TEXTS[i % len(TEXTS)] for i in range(bs)]
    return imgs, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--max", type=int, default=320)
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--method", default="cross_attn_alpha")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = ScheiCIR(method=args.method, temperature=0.07, backbone_size="L").to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    tokenizer = model.text_encoder.tokenizer

    results = []
    best = None
    for bs in range(args.start, args.max + 1, args.step):
        torch.cuda.reset_peak_memory_stats()
        try:
            imgs, texts = make_batch(bs, device, tokenizer)
            feats = model(imgs, texts, return_attention=False)
            loss = feats.norm()
            loss.backward()
            opt.zero_grad()
            peak = torch.cuda.max_memory_allocated() / 1e9
            results.append({"batch": bs, "ok": True, "peak_gb": round(peak, 2)})
            print(f"batch {bs}: OK, peak {peak:.2f} GB", flush=True)
            best = bs
        except torch.cuda.OutOfMemoryError:
            results.append({"batch": bs, "ok": False, "peak_gb": None})
            print(f"batch {bs}: OOM", flush=True)
            torch.cuda.empty_cache()
            break

    out = {"best_batch": best, "runs": results}
    with open("audit/vitl_batch_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"BEST STABLE BATCH: {best}")
    print("Wrote audit/vitl_batch_probe.json")


if __name__ == "__main__":
    main()
