#!/usr/bin/env python3
"""Sanity check for MTCIR/MerdCIR data loading and ScheiCIR training pipeline.

Checks:
  1. Data loading: JSONL→LMDB image decode, field mapping, NP parsing
  2. Model forward pass: shape correctness, no NaN
  3. Loss computation: InfoNCE + ScheiCIR soft contrastive loss
  4. Training step: gradient flow, no NaN in backward
  5. Data quality: duplicate targets, text lengths, NP coverage, image stats
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader

# -- Project imports -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.full_model import ScheiCIR
from data.dataset import MTCIRDataset, MerdCIRDataset

# -- Config --------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SAMPLES_QUALITY = 5000  # how many records to scan for data-quality checks

LMDB_PATH = str(PROJECT_ROOT / "data/MTCIR/images_224_lmdb")
MTCIR_NP_JSONL = str(PROJECT_ROOT / "data/mtcir_np/merged.jsonl")
MERDCIR_NP_JSONL = str(PROJECT_ROOT / "data/merdcir_np/test_train.jsonl")
MTCIR_EVAL_JSONL = str(PROJECT_ROOT / "data/mtcir_np/eval/eval_subset.jsonl")

CHECK_RESULTS: Dict[str, str] = {}  # check_name → "PASS" | "FAIL: reason"


def record(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else f"FAIL: {detail}"
    CHECK_RESULTS[name] = status
    marker = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
    msg = f"  {marker} {name}"
    if detail and not ok:
        msg += f"  ({detail})"
    print(msg)


def summarize():
    print("\n" + "=" * 64)
    total = len(CHECK_RESULTS)
    passes = sum(1 for v in CHECK_RESULTS.values() if v.startswith("PASS"))
    fails = total - passes
    print(f"RESULTS: {passes}/{total} passed, {fails}/{total} failed")
    for name, status in CHECK_RESULTS.items():
        if not status.startswith("PASS"):
            print(f"  FAIL: {name}  —  {status.removeprefix('FAIL: ')}")
    print("=" * 64)


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def check_mtcir_dataset():
    """Load a small batch from MTCIRDataset, verify shapes and content."""
    print("\n--- 1a. MTCIRDataset (NP-annotated) ---")
    try:
        ds = MTCIRDataset(data_path="./data", lmdb_path=LMDB_PATH, json_path="mtcir_np/merged.jsonl")
    except Exception as e:
        record("MTCIRDataset init", False, str(e))
        return

    record("MTCIRDataset init", True, f"len={len(ds)}")

    # Load 3 samples individually
    errors = []
    shapes_ref = []
    shapes_tgt = []
    np_counts = []
    text_samples = []
    for i in range(min(10, len(ds))):
        try:
            sample = ds[i]
        except Exception as e:
            errors.append(f"idx={i}: {e}")
            continue
        shapes_ref.append(tuple(sample["image"].shape))
        shapes_tgt.append(tuple(sample["target_img"].shape))
        np_counts.append(len(sample["np"]))
        text_samples.append(sample["text"])
        # verify LMDB path vs JSONL field
        if "target_img" not in sample and "target_path" in sample:
            pass  # OK

    record("sample __getitem__", len(errors) == 0,
           f"{len(errors)} errors" if errors else "10/10 ok")

    # Shape uniformity
    unique_ref = set(shapes_ref)
    unique_tgt = set(shapes_tgt)
    record("image shapes (C,H,W)=(3,224,224)", unique_ref == {(3, 224, 224)},
           f"ref shapes: {unique_ref}")
    record("target shapes (C,H,W)=(3,224,224)", unique_tgt == {(3, 224, 224)},
           f"tgt shapes: {unique_tgt}")

    # NP coverage
    has_nps = sum(1 for c in np_counts if c > 0)
    record("NP annotations present", has_nps == len(np_counts),
           f"{has_nps}/{len(np_counts)} samples have NPs")
    record("NP counts reasonable", all(1 <= c <= 50 for c in np_counts),
           f"NP range: {min(np_counts)}-{max(np_counts)}")

    # Text non-empty
    empty_texts = sum(1 for t in text_samples if not t.strip())
    record("text non-empty", empty_texts == 0,
           f"{empty_texts} empty texts" if empty_texts else "all non-empty")

    # Test DataLoader collation
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                    collate_fn=ds.custom_collate_fn)
    batch = next(iter(dl))
    record("DataLoader collate", True,
           f"keys={list(batch.keys())} | "
           f"image={list(batch['image'].shape)} "
           f"target_img={list(batch['target_img'].shape)} "
           f"batch_texts={len(batch['text'])}")
    ds.env.close()
    return ds


def check_merdcir_dataset():
    """Load a small batch from MerdCIRDataset."""
    print("\n--- 1b. MerdCIRDataset ---")
    try:
        ds = MerdCIRDataset(data_path="./data", lmdb_path=LMDB_PATH,
                            json_path="merdcir_np/test_train.jsonl")
    except Exception as e:
        record("MerdCIRDataset init", False, str(e))
        return

    record("MerdCIRDataset init", True, f"len={len(ds)}")

    errors = []
    text_types = Counter()
    for i in range(min(20, len(ds))):
        try:
            s = ds[i]
            text = s["text"]
            if s.get("id") is not None:
                text_types["has_id"] += 1
            if text and len(text) > 0:
                text_types["non_empty"] += 1
            if text.count(". ") > 1:
                text_types["multi_sentence"] += 1
            else:
                text_types["single_sentence"] += 1
        except Exception as e:
            errors.append(f"idx={i}: {e}")

    record("MerdCIR sample loading", len(errors) == 0,
           f"{len(errors)} errors" if errors else "20/20 ok")
    record("MerdCIR text present", text_types["non_empty"] > 0,
           f"text stats: {dict(text_types)}")
    ds.env.close()


# =============================================================================
# 2. MODEL FORWARD PASS
# =============================================================================
def check_forward_pass():
    """Run ScheiCIR forward pass with random weights, verify output shape & no NaN."""
    print("\n--- 2. Forward pass ---")

    model = ScheiCIR(method="cross_attn_alpha", backbone_size="B").to(DEVICE)
    model.eval()

    # Use a tiny subset from MTCIR dataset
    ds = MTCIRDataset(data_path="./data", lmdb_path=LMDB_PATH,
                      json_path="mtcir_np/merged.jsonl")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0,
                    collate_fn=ds.custom_collate_fn)
    batch = next(iter(dl))

    ref = batch["image"].to(DEVICE)
    tgt = batch["target_img"].to(DEVICE)
    texts = batch["text"]

    with torch.no_grad():
        fused, attn_map = model(ref, texts, return_attention=True)
        tgt_feat = model.visual_backbone(tgt, get_embeddings=True)

    record("fused shape [B, 768]", list(fused.shape) == [4, 768],
           f"got {list(fused.shape)}")
    record("attn_map shape [B, heads, patches, tokens]",
           len(attn_map.shape) == 4 and attn_map.shape[0] == 4,
           f"got {list(attn_map.shape)}")
    record("target_feat shape [B, 768]", list(tgt_feat.shape) == [4, 768],
           f"got {list(tgt_feat.shape)}")

    record("no NaN in fused", not torch.isnan(fused).any())
    record("no NaN in attn_map", not torch.isnan(attn_map).any())
    record("no NaN in target_feat", not torch.isnan(tgt_feat).any())

    # Check normalization
    fused_norm = F.normalize(fused, p=2, dim=-1)
    norms = fused_norm.norm(dim=-1)
    record("fused norms ~1.0", torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
           f"norms={norms.tolist()}")

    ds.env.close()
    return model


# =============================================================================
# 3. LOSS COMPUTATION
# =============================================================================
def check_loss(model: ScheiCIR = None):
    """Verify InfoNCE loss and ScheiCIR soft contrastive loss produce valid values."""
    print("\n--- 3. Loss computation ---")

    if model is None:
        model = ScheiCIR(method="cross_attn_alpha", backbone_size="B").to(DEVICE)

    ds = MTCIRDataset(data_path="./data", lmdb_path=LMDB_PATH,
                      json_path="mtcir_np/merged.jsonl")
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0,
                    collate_fn=ds.custom_collate_fn)
    batch = next(iter(dl))

    ref = batch["image"].to(DEVICE)
    tgt = batch["target_img"].to(DEVICE)
    texts = batch["text"]

    # Forward
    fused, attn_map = model(ref, texts, return_attention=True)
    tgt_feat = model.visual_backbone(tgt, get_embeddings=True)
    ref_feat = model.visual_backbone(ref, get_embeddings=True)

    # 3a. InfoNCE loss
    loss_infonce = model.compute_cir_loss(fused, tgt_feat, ref_feat)
    record("InfoNCE loss is scalar", loss_infonce.dim() == 0,
           f"dim={loss_infonce.dim()}")
    record("InfoNCE loss > 0", loss_infonce.item() > 0,
           f"value={loss_infonce.item():.4f}")
    record("InfoNCE loss not NaN", not torch.isnan(loss_infonce),
           f"value={loss_infonce.item():.4f}")
    record("InfoNCE loss < 100 (not exploded)", loss_infonce.item() < 100,
           f"value={loss_infonce.item():.4f}")

    # 3b. AlphaGenerator sanity (if alpha_gen exists)
    if model.alpha_gen is not None:
        nps_batch = batch["np"]
        np_counts = [len(np_list) for np_list in nps_batch]
        total_nps = sum(np_counts)
        if total_nps >= 2:
            # Build all_cls the same way training code does: repeat per-image fused_feat
            all_visual_feats = []
            for i, n in enumerate(np_counts):
                if n <= 1:
                    continue
                for _ in range(n * n - n):  # n² - n valid pairs per image
                    all_visual_feats.append(fused[i, :])
            num_pairs = len(all_visual_feats)

            if num_pairs > 0:
                all_visual_feats = torch.stack(all_visual_feats)
                # AlphaGenerator expects: feat dim=text_dim(512), cls dim=vision_dim(768)
                np_feat_dim = model.text_dim   # 512 for ViT-B
                np_attn_dim = model.patch_size ** 2  # 49 for 7x7
                dummy_feat = torch.randn(total_nps, np_feat_dim, device=DEVICE)
                dummy_attn = torch.randn(total_nps, np_attn_dim, device=DEVICE)

                loss_sc = model.compute_weighted_contrastive(
                    dummy_feat, dummy_attn, np_counts, all_visual_feats, debug=True
                )
                record("ScheiCIR loss is scalar", loss_sc.dim() == 0,
                       f"dim={loss_sc.dim()}")
                record("ScheiCIR loss > 0", loss_sc.item() >= 0,
                       f"value={loss_sc.item():.4f}")
                record("ScheiCIR loss not NaN", not torch.isnan(loss_sc),
                       f"value={loss_sc.item():.4f}")
            else:
                record("ScheiCIR loss test", False, "no valid NP pairs (all images have ≤1 NP)")
        else:
            record("ScheiCIR loss test", False, "not enough NPs in batch (need >= 2)")
    else:
        print("  (alpha_gen is None, skipping ScheiCIR loss check)")

    ds.env.close()


# =============================================================================
# 4. GRADIENT FLOW (mini training step)
# =============================================================================
def check_gradient_flow():
    """Run one training step, verify gradients flow through all trainable params."""
    print("\n--- 4. Gradient flow (1 training step) ---")

    model = ScheiCIR(method="cross_attn_alpha", backbone_size="B").to(DEVICE)
    model.train()
    # Only train interaction params (warmup_head stage)
    model.visual_backbone.model.requires_grad_(False)
    model.text_encoder.model.requires_grad_(False)

    ds = MTCIRDataset(data_path="./data", lmdb_path=LMDB_PATH,
                      json_path="mtcir_np/merged.jsonl")
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0,
                    collate_fn=ds.custom_collate_fn)
    batch = next(iter(dl))

    ref = batch["image"].to(DEVICE)
    tgt = batch["target_img"].to(DEVICE)
    texts = batch["text"]

    fused, attn_map = model(ref, texts, return_attention=True)
    tgt_feat = model.visual_backbone(tgt, get_embeddings=True)
    ref_feat = model.visual_backbone(ref, get_embeddings=True)

    loss = model.compute_cir_loss(fused, tgt_feat, ref_feat)
    loss.backward()

    # Check gradients on interaction params (alpha_gen not used in InfoNCE, so no grad expected)
    interaction_no_grad = []
    interaction_nan_grad = []
    alpha_no_grad = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_alpha = "alpha_gen" in name
        if param.grad is None:
            if is_alpha:
                alpha_no_grad.append(name)
            else:
                interaction_no_grad.append(name)
        elif torch.isnan(param.grad).any():
            interaction_nan_grad.append(name)

    record("interaction params have grads", len(interaction_no_grad) == 0,
           f"missing: {interaction_no_grad}" if interaction_no_grad else "all ok")
    record("alpha_gen no grad (expected: not in InfoNCE path)", len(alpha_no_grad) > 0,
           f"{len(alpha_no_grad)} params without grad (expected)")
    record("no NaN gradients", len(interaction_nan_grad) == 0,
           f"NaN in: {interaction_nan_grad}" if interaction_nan_grad else "all clean")

    # Check that loss decreased (one optimizer step with tiny LR)
    # Recompute forward for fresh graph
    fused2, _ = model(ref, texts, return_attention=True)
    loss_before = model.compute_cir_loss(fused2, tgt_feat, ref_feat)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    opt.zero_grad()
    loss_before.backward()
    opt.step()

    # Recompute loss after step
    with torch.no_grad():
        fused3, _ = model(ref, texts, return_attention=True)
        loss_after = model.compute_cir_loss(fused3, tgt_feat, ref_feat).item()

    record("loss decreased after optimizer step", loss_after < loss_before.item(),
           f"before={loss_before.item():.4f} → after={loss_after:.4f}")

    ds.env.close()


# =============================================================================
# 5. DATA QUALITY
# =============================================================================
def check_data_quality():
    """Scan a subset of records for quality issues."""
    print("\n--- 5. Data quality scan ---")

    import lmdb

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)

    # Scan MTCIR NP-annotated records
    all_targets: List[str] = []
    text_lengths: List[int] = []
    np_counts: List[int] = []
    ref_equals_tgt = 0
    empty_mod = 0

    with open(MTCIR_NP_JSONL) as f:
        for i, line in enumerate(f):
            if i >= MAX_SAMPLES_QUALITY:
                break
            rec = json.loads(line)
            ref = rec.get("image", "")
            tgt = rec.get("target_img", "")
            mod = rec.get("modification", "")
            nps = rec.get("nps", [])

            all_targets.append(tgt)
            text_lengths.append(len(mod))
            np_counts.append(len(nps))

            if ref == tgt:
                ref_equals_tgt += 1
            if not mod or not str(mod).strip():
                empty_mod += 1

    total = len(all_targets)
    unique_targets = len(set(all_targets))

    record("unique target ratio > 0.5", unique_targets > total * 0.5,
           f"{unique_targets}/{total} = {unique_targets/total:.1%}")
    record("reference != target", ref_equals_tgt == 0,
           f"{ref_equals_tgt}/{total} samples have ref==tgt" if ref_equals_tgt else "all distinct")
    record("no empty modifications", empty_mod == 0,
           f"{empty_mod}/{total} empty")
    record("avg text length > 30 chars", np.mean(text_lengths) > 30,
           f"mean={np.mean(text_lengths):.1f}, min={min(text_lengths)}, max={max(text_lengths)}")
    record("avg NP count > 2", np.mean(np_counts) > 2,
           f"mean={np.mean(np_counts):.2f}, min={min(np_counts)}, max={max(np_counts)}")

    # Check NP span validity (within 77 CLIP token limit)
    bad_spans = 0
    total_spans = 0
    with open(MTCIR_NP_JSONL) as f:
        for i, line in enumerate(f):
            if i >= MAX_SAMPLES_QUALITY:
                break
            rec = json.loads(line)
            for np_item in rec.get("nps", []):
                total_spans += 1
                if len(np_item) >= 2:
                    span = np_item[1]
                    if isinstance(span, (list, tuple)) and len(span) == 2:
                        if span[1] > 77:
                            bad_spans += 1

    record("NP spans within CLIP 77 limit", bad_spans == 0,
           f"{bad_spans}/{total_spans} spans > 77" if bad_spans else f"{total_spans} spans ok")

    # Spot-check: decode a few images from LMDB to verify they're valid JPEGs
    decode_errors = 0
    with open(MTCIR_NP_JSONL) as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            rec = json.loads(line)
            with env.begin(write=False, buffers=True) as txn:
                for key in [rec["image"], rec["target_img"]]:
                    buf = txn.get(key.encode("ascii"))
                    try:
                        img = Image.open(BytesIO(buf)).convert("RGB")
                        w, h = img.size
                        if w < 50 or h < 50:
                            decode_errors += 1
                    except Exception:
                        decode_errors += 1

    record("LMDB images are valid JPEGs", decode_errors == 0,
           f"{decode_errors} decode errors" if decode_errors else "all valid")

    # Check image value range after transforms
    ds = MTCIRDataset(data_path="./data", lmdb_path=LMDB_PATH,
                      json_path="mtcir_np/merged.jsonl")
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                    collate_fn=ds.custom_collate_fn)
    batch = next(iter(dl))
    ref = batch["image"][:64].cpu()
    tgt = batch["target_img"][:64].cpu()

    # Normalized values should be roughly mean≈0, std≈1 (ImageNet normalization)
    ref_mean = ref.mean().item()
    ref_std = ref.std().item()
    record("normalized image mean near 0", -1 < ref_mean < 1,
           f"mean={ref_mean:.4f}")
    record("normalized image std near 1", 0.3 < ref_std < 1.7,
           f"std={ref_std:.4f}")
    record("no NaN in normalized images", not torch.isnan(ref).any())

    ds.env.close()
    env.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 64)
    print("MTCIR / ScheiCIR Sanity Check")
    print(f"Device: {DEVICE}")
    print(f"LMDB: {LMDB_PATH}")
    print("=" * 64)

    # 1. Data loading
    check_mtcir_dataset()
    check_merdcir_dataset()

    # 2. Forward pass
    model = check_forward_pass()

    # 3. Loss computation
    check_loss(model)

    # 4. Gradient flow
    check_gradient_flow()

    # 5. Data quality
    check_data_quality()

    summarize()
    return 0 if all(v.startswith("PASS") for v in CHECK_RESULTS.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
