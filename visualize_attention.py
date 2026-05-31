#!/usr/bin/env python3
"""Visualize NP-level spatial attention maps for ScheiCIR CIR diagnostics.

The script follows the NP-attention extraction logic used during training:
1. Run ScheiCIR on the reference image and selected text.
2. Average the final image-token -> text-token cross-attention over heads.
3. For each noun phrase span, collapse its text-token attention into one spatial
   patch map with max over NP tokens.
4. Resize the patch map to the original reference image size and overlay it.

Example:
    python visualize_attention.py --id mock_000 --checkpoint checkpoints/topk_merdcir_cross_attn/topk_epoch_0003_step_000900_score_0.483130.pth.tar
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.full_model import ScheiCIR


DEFAULT_CHECKPOINT_GLOB = "checkpoints/topk*/*.pth.tar"
TEXT_KEYS = {
    "mtcir": ("mtcir_text", "modification", "modifications"),
    "merdcir": ("merdcir_text", "merdcir_modification", "modification", "modifications"),
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "into",
    "is", "it", "its", "make", "more", "no", "not", "of", "on", "or", "remove", "replace",
    "show", "switch", "the", "this", "to", "with", "without", "change", "add", "include",
    "keep", "move", "shift", "turn", "transform", "convert", "instead", "than",
}


NounPhrase = Tuple[str, Tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ScheiCIR NP-level spatial attention maps.")
    parser.add_argument("--id", dest="sample_id", help="Sample id to visualize. If omitted, prompts interactively.")
    parser.add_argument("--samples", default="samples.json", help="Path to samples JSON or JSONL file.")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to latest file under checkpoints/topk*/*.pth.tar.")
    parser.add_argument("--checkpoint-glob", default=DEFAULT_CHECKPOINT_GLOB, help="Glob used when --checkpoint is omitted.")
    parser.add_argument("--text-source", choices=("mtcir", "merdcir"), default="merdcir", help="Which text field to analyze.")
    parser.add_argument(
        "--method",
        default="auto",
        help="ScheiCIR method. Use 'auto' to infer from checkpoint path when possible.",
    )
    parser.add_argument("--backbone-size", choices=("B", "L", "H"), default="B", help="CLIP backbone size.")
    parser.add_argument("--num-cross-attn-layers", type=int, default=4, help="Number of ScheiCIR cross-attention layers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--out-dir", default="plots", help="Directory for saved figures.")
    parser.add_argument("--out", help="Explicit output PDF path.")
    parser.add_argument("--max-nps", type=int, default=10, help="Maximum noun phrases to visualize.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Reference image blend weight.")
    parser.add_argument("--beta", type=float, default=0.4, help="Heatmap blend weight.")
    parser.add_argument("--image-root", help="Optional root for relative image paths in samples.")
    parser.add_argument("--allow-random-model", action="store_true", help="Continue with random weights if no checkpoint is found.")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "samples", "annotations", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported sample file format: {path}")


def find_sample(records: Sequence[Dict[str, Any]], sample_id: Optional[str]) -> Dict[str, Any]:
    if sample_id is None:
        sample_id = input("Sample ID: ").strip()
    for row in records:
        if str(row.get("id")) == str(sample_id):
            return row
    raise KeyError(f"Sample id not found: {sample_id}")


def first_value(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return None


def text_from_sample(row: Dict[str, Any], text_source: str) -> str:
    value = first_value(row, TEXT_KEYS[text_source])
    if value is None:
        raise KeyError(f"No usable {text_source} text found in sample {row.get('id')}")
    if isinstance(value, list):
        return ". ".join(str(part).strip(". ") for part in value if str(part).strip())
    return str(value)


def image_path_from_sample(row: Dict[str, Any], kind: str, image_root: Optional[Path]) -> Path:
    if kind == "reference":
        keys = ("reference_path", "ref_path", "image", "reference", "reference_img")
    else:
        keys = ("target_path", "target_img", "target_image", "target")

    value = first_value(row, keys)
    if value is None:
        raise KeyError(f"No {kind} image path/key found in sample {row.get('id')}")

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        root = image_root if image_root is not None else Path.cwd()
        path = root / path
    if not path.exists():
        raise FileNotFoundError(
            f"{kind.title()} image not found: {path}\n"
            "If sample paths are dataset-relative keys, pass --image-root pointing to the image directory."
        )
    return path


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def find_checkpoint(path_arg: Optional[str], pattern: str) -> Optional[Path]:
    if path_arg:
        path = Path(path_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = [Path(p) for p in glob.glob(pattern)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not locate a model state_dict in the checkpoint.")


def strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def infer_method(method_arg: str, ckpt_path: Optional[Path]) -> str:
    if method_arg != "auto":
        return method_arg
    if ckpt_path is None:
        return "cross_attn_alpha"

    name = str(ckpt_path).lower()
    if "cross_attn" in name:
        return "cross_pooling_alpha"
    if "no_alpha" in name:
        return "cross_attn"
    return "cross_attn_alpha"


def load_model(args: argparse.Namespace) -> ScheiCIR:
    ckpt_path = find_checkpoint(args.checkpoint, args.checkpoint_glob)
    method = infer_method(args.method, ckpt_path)
    print(f"Using ScheiCIR method: {method}")
    model = ScheiCIR(
        method=method,
        backbone_size=args.backbone_size,
        num_cross_attn_layers=args.num_cross_attn_layers,
    )
    if ckpt_path is None:
        if not args.allow_random_model:
            raise FileNotFoundError(
                f"No checkpoint matched {args.checkpoint_glob}. Pass --checkpoint or --allow-random-model."
            )
        print("Warning: no checkpoint found; using randomly initialized model.", file=sys.stderr)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = strip_prefixes(checkpoint_state_dict(checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"Warning: {len(missing)} missing keys while loading checkpoint.", file=sys.stderr)
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys while loading checkpoint.", file=sys.stderr)

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    return model


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9'-]", "", text.lower())


def simple_np_candidates(text: str, max_nps: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    phrases: List[str] = []
    current: List[str] = []

    for token in tokens:
        norm = normalize_word(token)
        if norm and norm not in STOPWORDS:
            current.append(token)
            if len(current) >= 4:
                phrases.append(" ".join(current))
                current = []
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))

    unique = []
    seen = set()
    for phrase in phrases:
        key = normalize_word(phrase)
        if key and key not in seen:
            unique.append(phrase)
            seen.add(key)
        if len(unique) >= max_nps:
            break
    return unique or [text]


def token_span_for_phrase(model: ScheiCIR, text: str, phrase: str) -> Optional[Tuple[int, int]]:
    tokenizer = model.text_encoder.tokenizer
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    if not match:
        return None

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=77,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offsets = encoded.get("offset_mapping", [])
    token_indices = []
    for idx, offset in enumerate(offsets):
        if not offset or tuple(offset) == (0, 0):
            continue
        start, end = offset
        if end > match.start() and start < match.end():
            token_indices.append(idx)
    if not token_indices:
        return None
    # Keep the training convention: stored spans are 1-based inclusive positions,
    # then extraction uses start - 1:end over CLIP token indices.
    return token_indices[0] + 1, token_indices[-1] + 1


def nps_from_sample(row: Dict[str, Any], text: str, model: ScheiCIR, max_nps: int) -> List[NounPhrase]:
    raw_nps = row.get("nps") or row.get("np")
    parsed: List[NounPhrase] = []
    if isinstance(raw_nps, list):
        for item in raw_nps:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            phrase = str(item[0])
            span = item[1]
            if isinstance(span, (list, tuple)) and len(span) == 2:
                parsed.append((phrase, (int(span[0]), int(span[1]))))
            if len(parsed) >= max_nps:
                break
    if parsed:
        return parsed

    for phrase in simple_np_candidates(text, max_nps=max_nps):
        span = token_span_for_phrase(model, text, phrase)
        if span is not None:
            parsed.append((phrase, span))
    return parsed


def square_grid_size(num_patches: int) -> int:
    grid = int(math.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"Expected a square patch grid, got {num_patches} patches.")
    return grid


def extract_np_attention_maps(
    model: ScheiCIR,
    reference_image: Image.Image,
    text: str,
    nps: Sequence[NounPhrase],
) -> List[Tuple[str, np.ndarray]]:
    with torch.no_grad():
        _, attn_map = model([reference_image], [text], return_attention=True)
        avg_attn = attn_map.mean(dim=1)[0]  # [image_tokens, text_tokens]

    if avg_attn.shape[0] <= 1:
        raise ValueError(f"Attention map has no patch tokens: shape={tuple(avg_attn.shape)}")
    patch_attn = avg_attn[1:, :]  # Drop CLS image token, matching training code.
    grid = square_grid_size(patch_attn.shape[0])

    outputs: List[Tuple[str, np.ndarray]] = []
    for phrase, (start, end) in nps:
        if start > 77 or end > 77:
            print(f"Skipping NP beyond CLIP max length: {phrase} span={(start, end)}", file=sys.stderr)
            continue
        token_start = max(start - 1, 0)
        token_end = min(end, patch_attn.shape[1])
        if token_start >= token_end:
            print(f"Skipping invalid NP span: {phrase} span={(start, end)}", file=sys.stderr)
            continue

        token_attn = patch_attn[:, token_start:token_end].max(dim=1).values
        spatial = token_attn.reshape(grid, grid).detach().to("cpu").float().numpy()
        outputs.append((phrase, spatial))

    return outputs


def normalize_map(attn: np.ndarray) -> np.ndarray:
    attn = np.asarray(attn, dtype=np.float32)
    attn = attn - float(np.nanmin(attn))
    denom = float(np.nanmax(attn))
    if denom > 1e-8:
        attn = attn / denom
    return np.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)


def overlay_heatmap(reference_image: Image.Image, attn: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    rgb = np.asarray(reference_image.convert("RGB"))
    height, width = rgb.shape[:2]
    attn_norm = normalize_map(attn)
    resized = cv2.resize(attn_norm, (width, height), interpolation=cv2.INTER_LINEAR)
    heat_uint8 = np.uint8(np.clip(resized, 0.0, 1.0) * 255)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, alpha, heat_rgb, beta, 0)


def short_title(text: str, max_chars: int = 76) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def save_figure(
    sample_id: str,
    reference_image: Image.Image,
    target_image: Image.Image,
    text: str,
    overlays: Sequence[Tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    num_panels = 2 + len(overlays)
    cols = min(4, num_panels)
    rows = int(math.ceil(num_panels / cols))
    fig_width = 4.2 * cols
    fig_height = 4.3 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    flat_axes = axes.ravel()

    flat_axes[0].imshow(reference_image)
    flat_axes[0].set_title("Reference Image", fontsize=11)
    flat_axes[0].axis("off")

    flat_axes[1].imshow(target_image)
    flat_axes[1].set_title("Target Image", fontsize=11)
    flat_axes[1].axis("off")

    for ax, (phrase, overlay) in zip(flat_axes[2:], overlays):
        ax.imshow(overlay)
        ax.set_title(f"Attention for: '{short_title(phrase, 38)}'", fontsize=10)
        ax.axis("off")

    for ax in flat_axes[num_panels:]:
        ax.axis("off")

    fig.suptitle(f"Sample {sample_id} | {short_title(text, 110)}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.samples))
    sample = find_sample(records, args.sample_id)
    sample_id = str(sample.get("id"))
    text = text_from_sample(sample, args.text_source)
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    reference_path = image_path_from_sample(sample, "reference", image_root)
    target_path = image_path_from_sample(sample, "target", image_root)

    reference_image = load_rgb(reference_path)
    target_image = load_rgb(target_path)
    model = load_model(args)
    nps = nps_from_sample(sample, text, model, max_nps=args.max_nps)
    if not nps:
        raise ValueError(f"No noun phrases could be resolved for sample {sample_id}.")

    maps = extract_np_attention_maps(model, reference_image, text, nps)
    if not maps:
        raise ValueError(f"No valid attention maps could be extracted for sample {sample_id}.")

    overlays = [
        (phrase, overlay_heatmap(reference_image, attn, alpha=args.alpha, beta=args.beta))
        for phrase, attn in maps
    ]
    out_path = Path(args.out) if args.out else Path(args.out_dir) / f"attention_fail_case_{sample_id}.pdf"
    save_figure(sample_id, reference_image, target_image, text, overlays, out_path)

    print(f"Saved attention visualization: {out_path}")
    print("Noun phrases:")
    for phrase, span in nps[: len(maps)]:
        print(f"  - {phrase} {span}")


if __name__ == "__main__":
    main()
