#!/usr/bin/env python3
"""Render reference/target contact sheets for the CIRR annotation queue."""
from __future__ import annotations

import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import lmdb
from PIL import Image, ImageDraw, ImageFont
from PIL import JpegImagePlugin  # noqa: F401  # register Pillow's PDF JPEG handler
from io import BytesIO


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "narrative_followup/annotations/cirr_mechanism_candidates.csv"
LMDB_PATH = ROOT / "data/CIRR/images_224_lmdb"
OUT = ROOT / "narrative_followup/annotations/contact_sheets"
ROWS_PER_PAGE = 5
THUMB = 160
PAGE_W = 920
ROW_H = 190


def font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def candidate_keys(key: str) -> list[str]:
    normalized = key.strip("./")
    suffix = Path(normalized).suffix
    candidates = [normalized]
    if not suffix:
        candidates.extend(f"{normalized}{ext}" for ext in (".png", ".jpg", ".jpeg"))
    if "/" not in normalized:
        for split in ("test1", "dev", "train"):
            if normalized.startswith(f"{split}-"):
                candidates.append(f"{split}/{normalized}")
                if not suffix:
                    candidates.extend(f"{split}/{normalized}{ext}" for ext in (".png", ".jpg", ".jpeg"))
                break
    return list(dict.fromkeys(candidates))


def read_image(txn, key: str) -> Image.Image:
    value = None
    for candidate in candidate_keys(key):
        value = txn.get(candidate.encode("ascii"))
        if value is not None:
            break
    if value is None:
        raise KeyError(f"CIRR LMDB image missing for display id: {key}")
    with Image.open(BytesIO(bytes(value))) as image:
        return image.convert("RGB")


def fit(image: Image.Image) -> Image.Image:
    image = image.copy()
    image.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (THUMB, THUMB), "white")
    canvas.paste(image, ((THUMB - image.width) // 2, (THUMB - image.height) // 2))
    return canvas


def make_page(rows, txn, group: str, page_number: int):
    title_font = font(22)
    body_font = font(15)
    small_font = font(12)
    page = Image.new("RGB", (PAGE_W, ROW_H * ROWS_PER_PAGE + 55), "white")
    draw = ImageDraw.Draw(page)
    draw.text((12, 12), f"CIRR annotation: {group} — page {page_number}", fill="black", font=title_font)
    for idx, row in enumerate(rows):
        y = 50 + idx * ROW_H
        draw.line((8, y - 3, PAGE_W - 8, y - 3), fill=(215, 215, 215), width=1)
        ref = fit(read_image(txn, row["reference"]))
        target = fit(read_image(txn, row["target"]))
        page.paste(ref, (12, y))
        page.paste(target, (190, y))
        draw.text((14, y + THUMB + 2), "reference", fill=(70, 70, 70), font=small_font)
        draw.text((192, y + THUMB + 2), "target", fill=(70, 70, 70), font=small_font)
        x = 375
        draw.text((x, y + 2), f"pairid={row['pairid']}", fill="black", font=body_font)
        draw.text((x, y + 25), f"global ranks F/M: {row['fixed_global_rank']}/{row['multi_global_rank']}", fill="black", font=small_font)
        draw.text((x, y + 44), f"subset ranks F/M: {row['fixed_subset_rank']}/{row['multi_subset_rank']}", fill="black", font=small_font)
        caption = textwrap.fill(row["caption"], width=57)
        draw.multiline_text((x, y + 68), caption, fill=(25, 25, 25), font=small_font, spacing=3)
    return page


def main() -> None:
    rows_by_group = defaultdict(list)
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_group[row["annotation_group"]].append(row)
    if not rows_by_group:
        raise SystemExit("annotation CSV is empty")
    if not LMDB_PATH.is_dir():
        raise SystemExit(f"missing CIRR LMDB: {LMDB_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove only known generated group PDFs so a regenerated queue cannot
    # leave stale strata alongside the current contact sheets.
    for stale_name in (
        "fixed_global_win.pdf", "fixed_subset_win.pdf", "multi_global_win.pdf",
        "multi_subset_loss.pdf", "fixed_win.pdf", "stratified_random.pdf",
    ):
        stale = OUT / stale_name
        if stale.exists():
            stale.unlink()
    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, readahead=False, max_readers=32)
    manifest = []
    with env.begin(buffers=True) as txn:
        for group in sorted(rows_by_group):
            group_rows = rows_by_group[group]
            pages = [
                make_page(group_rows[i : i + ROWS_PER_PAGE], txn, group, i // ROWS_PER_PAGE + 1)
                for i in range(0, len(group_rows), ROWS_PER_PAGE)
            ]
            out_path = OUT / f"{group}.pdf"
            pages[0].save(out_path, save_all=True, append_images=pages[1:], resolution=120.0)
            manifest.append({"group": group, "rows": len(group_rows), "pages": len(pages), "file": str(out_path.relative_to(ROOT))})
    env.close()
    (OUT / "README.md").write_text(
        "# CIRR mechanism annotation contact sheets\n\n"
        "These sheets pair the reference and target images for the 600-row CSV. "
        "They are annotation aids only; the CSV's manual fields are intentionally blank.\n\n"
        + "\n".join(f"- `{item['group']}`: {item['rows']} rows, {item['pages']} pages — `{item['file']}`" for item in manifest)
        + "\n",
        encoding="utf-8",
    )
    print({"output": str(OUT), "groups": manifest})


if __name__ == "__main__":
    main()
