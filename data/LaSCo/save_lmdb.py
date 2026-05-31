import argparse
import os
from io import BytesIO

import lmdb
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    # Reproduction use:
    #   python data/LaSCo/save_lmdb.py \
    #       --image-root data/LaSCo/images \
    #       --output-lmdb data/LaSCo/images_224_lmdb
    #
    # Expected input is a COCO-style image tree, commonly containing folders
    # such as `train2014/` and `val2014/`. The generated LMDB is local data and
    # should not be committed.
    parser = argparse.ArgumentParser(
        description="Store LaSCo/COCO images as resized JPEG bytes in LMDB."
    )
    parser.add_argument(
        "--image-root",
        default="data/LaSCo/images",
        help="Root containing COCO folders, e.g. train2014/COCO_train2014_*.jpg.",
    )
    parser.add_argument(
        "--output-lmdb",
        default="data/LaSCo/images_224_lmdb",
        help="Output LMDB directory.",
    )
    parser.add_argument("--map-size-gb", type=int, default=50)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def iter_images(image_root):
    for root, _, files in os.walk(image_root):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, name)


def encode_image(path, quality):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality)
        return out.getvalue()


def key_for(path, image_root):
    # The key is the path relative to the image root. Keep the same image root
    # during evaluation so caption/image metadata can resolve the stored bytes.
    return os.path.relpath(path, image_root).replace("\\", "/").encode("ascii")


def create_lmdb(image_root, output_lmdb, map_size_gb=50, quality=90, batch_size=1024):
    # `map_size_gb` is an upper bound reserved by LMDB, not the immediate disk
    # size of the database. Increase it if a larger local image set raises
    # MDB_MAP_FULL.
    image_paths = sorted(iter_images(image_root))
    print(f"Found images: {len(image_paths)}")
    print(f"Image root: {image_root}")
    print(f"Output LMDB: {output_lmdb}")

    os.makedirs(output_lmdb, exist_ok=True)
    env = lmdb.open(output_lmdb, map_size=map_size_gb * 1024 * 1024 * 1024)

    written = 0
    skipped = 0
    txn = env.begin(write=True)
    try:
        for idx, path in enumerate(tqdm(image_paths, desc="Writing LMDB"), start=1):
            try:
                txn.put(key_for(path, image_root), encode_image(path, quality))
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"\nSkipping {path}: {exc}")

            if idx % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()

    print(f"Written images: {written}")
    print(f"Skipped images: {skipped}")


def main():
    args = parse_args()
    create_lmdb(
        image_root=args.image_root,
        output_lmdb=args.output_lmdb,
        map_size_gb=args.map_size_gb,
        quality=args.quality,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
