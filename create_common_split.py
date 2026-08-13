"""
Create common-dev split at triplet level.
In composed image retrieval, the same image with different modification text
constitutes a different task. This is consistent with CIRR/FashionIQ practice.
"""
import json
import os
import hashlib
import random

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data/pair_control")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 114514

def load_jsonl(path):
    data = {}
    for line in open(path):
        rec = json.loads(line)
        tid = rec.get("id", "")
        if tid:
            data[tid] = rec
    return data

def main():
    raw = load_jsonl(os.path.join(BASE, "data/mtcir_np/merged.jsonl"))
    fixed = load_jsonl(os.path.join(BASE, "data/fixed_prompt/full_train.jsonl"))
    multi = load_jsonl(os.path.join(BASE, "data/multi_intent/full_train.jsonl"))
    
    with open(os.path.join(OUT_DIR, "strict_intersection_ids.txt")) as f:
        strict_ids = sorted(set(line.strip() for line in f))
    
    print(f"Strict intersection: {len(strict_ids):,} triplets")
    
    # Shuffle with fixed seed
    rng = random.Random(SEED)
    rng.shuffle(strict_ids)
    
    DEV_SIZE = 10000
    TEST_SIZE = 10000
    
    dev_ids = set(strict_ids[:DEV_SIZE])
    test_ids = set(strict_ids[DEV_SIZE:DEV_SIZE + TEST_SIZE])
    train_ids = set(strict_ids[DEV_SIZE + TEST_SIZE:])
    
    print(f"Train: {len(train_ids):,}  Dev: {len(dev_ids):,}  Test: {len(test_ids):,}")
    
    # Check image overlap between splits (for transparency)
    def get_ref_imgs(ids, data):
        return set(data[tid].get("image", "") for tid in ids if tid in data)
    
    train_refs = get_ref_imgs(train_ids, raw)
    dev_refs = get_ref_imgs(dev_ids, raw)
    test_refs = get_ref_imgs(test_ids, raw)
    
    print(f"Reference image overlap:")
    print(f"  Train ∩ Dev:  {len(train_refs & dev_refs)}")
    print(f"  Train ∩ Test: {len(train_refs & test_refs)}")
    print(f"  Dev ∩ Test:   {len(dev_refs & test_refs)}")
    print(f"  (Note: reference image overlap is expected and consistent with CIRR/FashionIQ practice)")
    
    # Write splits
    def write_split(ids, fname, data_dict):
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            for tid in sorted(ids):
                if tid in data_dict:
                    f.write(json.dumps(data_dict[tid], ensure_ascii=False) + "\n")
        count = sum(1 for _ in open(path))
        print(f"  {fname}: {count} records")
    
    print("\nWriting splits...")
    write_split(train_ids, "train.jsonl", raw)
    write_split(dev_ids, "dev.jsonl", raw)
    write_split(test_ids, "test.jsonl", raw)
    write_split(train_ids, "train_fixed.jsonl", fixed)
    write_split(train_ids, "train_multi.jsonl", multi)
    write_split(dev_ids, "dev_fixed.jsonl", fixed)
    write_split(dev_ids, "dev_multi.jsonl", multi)
    write_split(test_ids, "test_fixed.jsonl", fixed)
    write_split(test_ids, "test_multi.jsonl", multi)
    
    # Save split IDs for reproducibility
    with open(os.path.join(OUT_DIR, "train_ids.txt"), "w") as f:
        for tid in sorted(train_ids): f.write(tid + "\n")
    with open(os.path.join(OUT_DIR, "dev_ids.txt"), "w") as f:
        for tid in sorted(dev_ids): f.write(tid + "\n")
    with open(os.path.join(OUT_DIR, "test_ids.txt"), "w") as f:
        for tid in sorted(test_ids): f.write(tid + "\n")
    
    # Manifest
    manifest = {
        "created": "2026-08-08",
        "split_method": "triplet-level random shuffle (consistent with CIRR/FashionIQ)",
        "seed": SEED,
        "strict_intersection_size": len(strict_ids),
        "splits": {
            "train": len(train_ids),
            "dev": len(dev_ids),
            "test": len(test_ids),
        },
        "reference_image_overlap": {
            "train_dev": len(train_refs & dev_refs),
            "train_test": len(train_refs & test_refs),
            "dev_test": len(dev_refs & test_refs),
        },
        "files": {},
    }
    
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        if not fname.endswith(".jsonl"):
            continue
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        manifest["files"][fname] = {
            "sha256": sha.hexdigest(),
            "lines": sum(1 for _ in open(fpath)),
        }
    
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nManifest: {os.path.join(OUT_DIR, 'manifest.json')}")
    print("Done!")

if __name__ == "__main__":
    main()
