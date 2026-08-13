#!/usr/bin/env python3
"""Smoke test: one real training step with ScheiCIR(ViT-L/14) on pair_control
train data to confirm no OOM with optimizer state + attention maps before
launching the full Fixed-L training run.
Mirrors train.py training-step memory profile (fused+attn forward, InfoNCE,
backward, optimizer step); SC-loss index ops are negligible vs. these.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.full_model import ScheiCIR
from data.dataset import MerdCIRDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--json-path", default="pair_control/train_fixed.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = ScheiCIR(method="cross_attn_alpha", temperature=0.07, backbone_size="L").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    dataset = MerdCIRDataset(
        data_path="./data",
        lmdb_path="./data/MTCIR/images_224_lmdb",
        json_path=args.json_path,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, collate_fn=dataset.custom_collate_fn)
    batch = next(iter(loader))
    ref_imgs = batch["image"].to(device)
    target_imgs = batch["target_img"].to(device)
    texts = batch["text"]

    torch.cuda.reset_peak_memory_stats()
    model.train()
    model.use_checkpoint = True
    fused_feat, attn_map = model(ref_imgs, texts, return_attention=True)
    from torch.utils.checkpoint import checkpoint as _ckpt
    target_feat = _ckpt(
        lambda imgs: model.visual_backbone(imgs, get_embeddings=True),
        target_imgs,
        use_reentrant=False,
    )
    ref_feat = _ckpt(
        lambda imgs: model.visual_backbone(imgs, get_embeddings=True),
        ref_imgs,
        use_reentrant=False,
    )
    infonce_loss = model.compute_cir_loss(fused_feat, target_feat, ref_feat)
    infonce_loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"fused {tuple(fused_feat.shape)}, attn {tuple(attn_map.shape)}")
    print(f"infonce loss: {infonce_loss.item():.4f}, peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"SMOKE TEST PASS (batch {args.batch_size})")


if __name__ == "__main__":
    main()
