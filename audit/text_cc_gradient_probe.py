#!/usr/bin/env python3
"""Matched-gradient probe for the Text CC loss (plan §8.8).

For each (stage, epsilon) cell, with identical initialization, batch and forward
outputs, we backward L_main, unweighted L_cc and lambda*L_cc separately and
record gradient norms over the fusion/shared parameters
(cross_attn_layers + attn_pooler):

    rho_g     = ||grad(lambda*L_cc)|| / ||grad(L_main)||
    cosine    = cos(grad(L_main), grad(lambda*L_cc))
    nonzero   = fraction of batch samples whose thresholded Text CC > 0

Epsilon levels: 0.0 (raw), paper/default (0.05), and the thesis-set 0.08.
Stages: samples from the beginning / middle / end of the CIRR train split to
approximate early / mid / late training data exposure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "reproduction_evidence"
sys.path.insert(0, str(SRC))

from Dataset import CIRDataset, collate_fn_with_nps  # noqa: E402
from Models import ConText  # noqa: E402


def build_model(args):
    model_args = argparse.Namespace(backbone_size=args.backbone_size, learning_rate=1e-5)
    model = ConText(
        args=model_args,
        num_cross_attn_layers=4,
        heads=8,
        dropout=0.0,
        weight_decay=1e-2,
        temperature=0.07,
        steps_per_epoch=1,
        max_epochs=1,
        warmup_steps=500,
        lambda_cc=args.lambda_cc,
        epsilon_cc=args.epsilon_cc,
        max_nps=args.max_nps,
        use_residual_fusion=True,
        init_parser=False,
    )
    return model.to(args.device)


def build_dataset(model, args):
    preproc = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
        ]
    )
    return CIRDataset(
        data_path=str(ROOT / "CIRR"),
        split=args.split,
        dataset="cirr",
        preprocess=preproc,
        parser=model.parser,
        max_nps=args.max_nps,
        np_cache_dir=str(OUT / "np_cache"),
        lazy_np_extraction=True,
    )


def fusion_params(model):
    names, params = [], []
    for name, p in model.named_parameters():
        if name.startswith("cross_attn_layers") or name.startswith("attn_pooler"):
            if p.requires_grad:
                names.append(name)
                params.append(p)
    return names, params


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def grad_flat(names, params):
    flat = []
    for name, p in zip(names, params):
        if p.grad is not None:
            flat.append(p.grad.detach().float().flatten())
        else:
            flat.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(flat) if flat else torch.zeros(1, device="cpu")


def cosine(a, b):
    if a.numel() == 0 or b.numel() == 0 or a.norm() == 0 or b.norm() == 0:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def collect_batch(dataset, indices, size=8):
    samples = []
    for idx in indices:
        sample = dataset[idx]
        if sample is None or not sample.get("noun_phrases"):
            continue
        samples.append(sample)
        if len(samples) == size:
            break
    if not samples:
        return None
    return collate_fn_with_nps(samples)


def measure(model, batch, device, epsilon, lambda_cc):
    """Return gradient stats for one (stage, epsilon) cell."""
    model.epsilon_cc = epsilon
    model.zero_grad(set_to_none=True)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    text_features = model.text_encoder(batch["caption"])
    vision_features = model.vision_encoder(batch["reference"])
    vl_embeddings, full_attention_maps = model.fuse_features(
        vision_features, text_features, return_attention=True
    )
    target_embeddings = model.vision_encoder(batch["target"], get_embeddings=True)
    query_embeddings = model.vision_encoder(batch["reference"], get_embeddings=True)
    all_target_embeddings = torch.cat([target_embeddings, query_embeddings], dim=0)
    vl_doubled = torch.cat([vl_embeddings, vl_embeddings], dim=0)
    loss_main = model.infonce(vl_doubled, all_target_embeddings)

    loss_cc = model.compute_text_cc_loss(
        batch, full_attention_maps=full_attention_maps, vision_features=vision_features
    )

    names, params = fusion_params(model)

    # g_main
    model.zero_grad(set_to_none=True)
    loss_main.backward(retain_graph=True)
    g_main = grad_norm(params)
    flat_main = grad_flat(names, params)

    # g_cc (unweighted)
    model.zero_grad(set_to_none=True)
    loss_cc.backward(retain_graph=True)
    g_cc = grad_norm(params)
    flat_cc = grad_flat(names, params)

    # g_lambda_cc
    model.zero_grad(set_to_none=True)
    (lambda_cc * loss_cc).backward()
    g_lambda_cc = grad_norm(params)
    flat_lambda = grad_flat(names, params)

    n_batch = len(batch["noun_phrases"])
    nonzero_cc = int(loss_cc.item() > 0) if loss_cc.ndim == 0 else -1

    return {
        "epsilon": epsilon,
        "loss_main": float(loss_main.item()),
        "loss_cc": float(loss_cc.item()),
        "weighted_loss_cc": float(lambda_cc * loss_cc.item()),
        "grad_norm_main": g_main,
        "grad_norm_cc": g_cc,
        "grad_norm_lambda_cc": g_lambda_cc,
        "rho_g": g_lambda_cc / g_main if g_main > 0 else float("inf"),
        "grad_ratio_unweighted": g_cc / g_main if g_main > 0 else float("inf"),
        "cosine_main_lambda_cc": cosine(flat_main, flat_lambda),
        "cosine_main_cc": cosine(flat_main, flat_cc),
        "batch_samples": n_batch,
        "loss_cc_nonzero": nonzero_cc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_nps", type=int, default=3)
    ap.add_argument("--lambda_cc", type=float, default=0.08)
    ap.add_argument("--epsilon_cc", type=float, default=0.05)
    ap.add_argument("--backbone_size", default="B", choices=["B", "L", "H"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--checkpoint", default=None,
                    help="Optional converged .ckpt (Lightning) to load instead of untrained init")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_prefix", default="text_cc_gradient_probe")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build_model(args)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing sample:", list(missing)[:5])
    model = model.to(args.device)
    dataset = build_dataset(model, args)
    n = len(dataset)
    print(f"dataset size: {n}")

    # early / mid / late stages
    stage_indices = {
        "early": list(range(0, 64)),
        "mid": list(range(n // 2, n // 2 + 64)),
        "late": list(range(n - 64, n)),
    }
    epsilons = [0.0, args.epsilon_cc, 0.08]

    results = {"stages": {}, "params_tracked": "cross_attn_layers + attn_pooler (fusion/shared)"}
    for stage, indices in stage_indices.items():
        batch = collect_batch(dataset, indices, size=args.batch_size)
        if batch is None:
            print(f"  {stage}: no usable samples, skipping")
            continue
        print(f"  stage={stage}: batch samples={len(batch['noun_phrases'])}")
        results["stages"][stage] = []
        for eps in epsilons:
            row = measure(model, batch, args.device, eps, args.lambda_cc)
            results["stages"][stage].append(row)
            print(f"    eps={eps}: loss_main={row['loss_main']:.4f} loss_cc={row['loss_cc']:.6f} "
                  f"rho_g={row['rho_g']:.3e} cosine={row['cosine_main_lambda_cc']:.4f}")

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"{args.output_prefix}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
