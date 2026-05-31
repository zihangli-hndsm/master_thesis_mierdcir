import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import lmdb
from PIL import Image, ImageOps
from tqdm import tqdm


DEFAULT_CATEGORIES = ("dress", "shirt", "toptee")


def parse_args():
    # Reproduction use:
    #   cd data/FashionIQ
    #   python save_lmdb.py --data-root . --split val
    #   python save_lmdb.py --data-root . --split test
    #
    # Required local inputs:
    #   captions/cap.{category}.{split}.json
    #   image_splits/split.{category}.{split}.json
    #   images/asin2url.{category}.txt
    #
    # The script downloads only ASINs used by the requested split and stores
    # 224x224 JPEG bytes in `images_{split}_224_lmdb/`. These LMDB folders and
    # the URL text files are treated as local data artifacts by `.gitignore`.
    parser = argparse.ArgumentParser(
        description="Download FashionIQ split images from asin2url files and store resized JPEGs in LMDB."
    )
    parser.add_argument("--data-root", default=".", help="FashionIQ root containing captions/, image_splits/, images/.")
    parser.add_argument("--split", default="test", help="FashionIQ split to materialize, e.g. test or val.")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=DEFAULT_CATEGORIES)
    parser.add_argument("--output-lmdb", default=None, help="Output LMDB path. Defaults to images_{split}_224_lmdb.")
    parser.add_argument("--map-size-gb", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--failures-json", default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_required_asins(data_root, split, categories):
    # Collect image ids from both split lists and caption pairs. This prevents
    # missing target/candidate images when evaluation uses captions rather than
    # only the image split file.
    required = set()
    for category in categories:
        split_path = os.path.join(data_root, "image_splits", f"split.{category}.{split}.json")
        cap_path = os.path.join(data_root, "captions", f"cap.{category}.{split}.json")

        if os.path.isfile(split_path):
            required.update(str(asin) for asin in load_json(split_path))

        if os.path.isfile(cap_path):
            for item in load_json(cap_path):
                for key in ("candidate", "target"):
                    if item.get(key):
                        required.add(str(item[key]))
    return required


def load_url_map(data_root, categories):
    # FashionIQ distributes ASIN-to-URL mappings by category. If these files are
    # missing, download or restore them before running this script; generated
    # LMDBs cannot be reproduced from the split JSON files alone.
    url_map = {}
    for category in categories:
        path = os.path.join(data_root, "images", f"asin2url.{category}.txt")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                asin, url = parts[0], parts[-1]
                url_map.setdefault(asin, url)
    return url_map


def key_exists(env, asin):
    # Makes the downloader resumable. Re-running the command skips images that
    # are already present in the output LMDB.
    with env.begin(write=False, buffers=True) as txn:
        return txn.get(asin.encode("ascii")) is not None


def encode_image(raw, quality):
    # EXIF transpose preserves the visual orientation used by the original
    # image. Every stored image is normalized to RGB 224x224 JPEG bytes.
    with Image.open(BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224))
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality)
        return out.getvalue()


def download_one(asin, url, timeout, retries, quality):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FashionIQDownloader/1.0)",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return asin, encode_image(raw, quality), None
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, Image.UnidentifiedImageError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return asin, None, last_error


def write_batch(env, batch):
    # Batch writes keep LMDB transactions reasonably small while avoiding one
    # transaction per image, which is slow for large FashionIQ splits.
    with env.begin(write=True) as txn:
        for asin, value in batch:
            txn.put(asin.encode("ascii"), value)


def main():
    args = parse_args()
    # Default output names match DATASET_PATHS in `eval_checkpoint.py`:
    # `images_val_224_lmdb` and `images_test_224_lmdb`.
    output_lmdb = args.output_lmdb or os.path.join(args.data_root, f"images_{args.split}_224_lmdb")
    required_asins = load_required_asins(args.data_root, args.split, args.categories)
    url_map = load_url_map(args.data_root, args.categories)
    missing_urls = sorted(asin for asin in required_asins if asin not in url_map)

    os.makedirs(output_lmdb, exist_ok=True)
    env = lmdb.open(output_lmdb, map_size=args.map_size_gb * 1024 * 1024 * 1024)
    pending = [(asin, url_map[asin]) for asin in sorted(required_asins) if asin in url_map and not key_exists(env, asin)]

    print(f"Required ASINs: {len(required_asins)}")
    print(f"Missing URLs: {len(missing_urls)}")
    print(f"Already in LMDB: {len(required_asins) - len(missing_urls) - len(pending)}")
    print(f"Pending downloads: {len(pending)}")
    print(f"Output LMDB: {output_lmdb}")

    failures = {asin: "missing URL" for asin in missing_urls}
    write_buffer = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, asin, url, args.timeout, args.retries, args.quality)
            for asin, url in pending
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            asin, value, error = future.result()
            if error:
                failures[asin] = error
                continue
            write_buffer.append((asin, value))
            if len(write_buffer) >= 128:
                write_batch(env, write_buffer)
                write_buffer.clear()

    if write_buffer:
        write_batch(env, write_buffer)

    env.sync()
    env.close()

    failures_json = args.failures_json or os.path.join(args.data_root, f"download_failures_{args.split}.json")
    with open(failures_json, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    print(f"Failures: {len(failures)}")
    print(f"Failure log: {failures_json}")


if __name__ == "__main__":
    main()
