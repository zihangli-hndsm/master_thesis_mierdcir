"""
Pair control audit: 验证 Raw(MTCIR)/Fixed/Multi 之间的 triplet 重叠情况。
输出:
  1. 各数据集的 triplet ID 数量
  2. 两两交集大小
  3. 三方严格交集大小
  4. reference--target mapping 一致性检查
  5. 提取匹配的 Raw 文本子集
"""
import json
import hashlib
import os
from collections import defaultdict

def load_jsonl(path):
    """Load a JSONL file and index by triplet ID."""
    data = {}
    for line in open(path):
        rec = json.loads(line)
        tid = rec.get("id", None)
        if tid is None:
            continue
        data[tid] = rec
    return data

def triplet_key(rec):
    """Return (reference_img, target_img) pair."""
    return (rec.get("image", ""), rec.get("target_img", ""))

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    raw_path = os.path.join(base, "data/mtcir_np/merged.jsonl")
    fixed_path = os.path.join(base, "data/fixed_prompt/full_train.jsonl")
    multi_path = os.path.join(base, "data/multi_intent/full_train.jsonl")
    
    print("=" * 60)
    print("PAIR CONTROL AUDIT")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading datasets...")
    raw = load_jsonl(raw_path)
    fixed = load_jsonl(fixed_path)
    multi = load_jsonl(multi_path)
    
    print(f"  Raw (MTCIR):   {len(raw):,} triplets")
    print(f"  Fixed prompt:  {len(fixed):,} triplets")
    print(f"  Multi intent:  {len(multi):,} triplets")
    
    # ID sets
    raw_ids = set(raw.keys())
    fixed_ids = set(fixed.keys())
    multi_ids = set(multi.keys())
    
    # Pair-wise intersections
    print("\n[2] Pair-wise ID overlap:")
    rf = raw_ids & fixed_ids
    rm = raw_ids & multi_ids
    fm = fixed_ids & multi_ids
    rfm = raw_ids & fixed_ids & multi_ids
    
    print(f"  Raw ∩ Fixed:  {len(rf):,} ({100*len(rf)/len(fixed):.1f}% of Fixed)")
    print(f"  Raw ∩ Multi:  {len(rm):,} ({100*len(rm)/len(multi):.1f}% of Multi)")
    print(f"  Fixed ∩ Multi: {len(fm):,} ({100*len(fm)/len(fixed):.1f}% of Fixed)")
    print(f"  Raw ∩ Fixed ∩ Multi (STRICT): {len(rfm):,}")
    
    # Check Fixed vs Multi consistency
    print("\n[3] Fixed vs Multi reference--target mapping consistency:")
    fm_mismatch = 0
    for tid in fm:
        if triplet_key(fixed[tid]) != triplet_key(multi[tid]):
            fm_mismatch += 1
    print(f"  Mismatched (image, target_img) pairs: {fm_mismatch} / {len(fm)}")
    
    # Check Raw vs Fixed/Multi mapping consistency
    print("\n[4] Raw vs Rewritten reference--target mapping consistency:")
    raw_vs_fixed_mismatch = 0
    for tid in rfm:
        if triplet_key(raw[tid]) != triplet_key(fixed[tid]):
            raw_vs_fixed_mismatch += 1
    print(f"  Raw vs Fixed mismatches in strict intersection: {raw_vs_fixed_mismatch} / {len(rfm)}")
    
    # Image-level overlap analysis
    print("\n[5] Image-level analysis (reference images):")
    raw_ref_imgs = set()
    fixed_ref_imgs = set()
    multi_ref_imgs = set()
    for tid in raw_ids:
        raw_ref_imgs.add(raw[tid].get("image", ""))
    for tid in fixed_ids:
        fixed_ref_imgs.add(fixed[tid].get("image", ""))
    for tid in multi_ids:
        multi_ref_imgs.add(multi[tid].get("image", ""))
    
    # Check for image leakage: images that appear in multiple triplets
    # This is important for preventing train/dev/test leakage
    print(f"  Unique reference images - Raw: {len(raw_ref_imgs):,}, Fixed: {len(fixed_ref_imgs):,}, Multi: {len(multi_ref_imgs):,}")
    
    # Check for duplicate images across triplets (same image in different triplets = need component-based split)
    raw_ref_counts = defaultdict(int)
    for tid in raw_ids:
        raw_ref_counts[raw[tid].get("image", "")] += 1
    fixed_ref_counts = defaultdict(int)
    for tid in fixed_ids:
        fixed_ref_counts[fixed[tid].get("image", "")] += 1
    
    raw_multi_triplet_imgs = sum(1 for c in raw_ref_counts.values() if c > 1)
    fixed_multi_triplet_imgs = sum(1 for c in fixed_ref_counts.values() if c > 1)
    print(f"  Images appearing in >1 triplet - Raw: {raw_multi_triplet_imgs}, Fixed: {fixed_multi_triplet_imgs}")
    
    # Text statistics for the strict intersection
    print("\n[6] Text statistics for strict intersection ({:,} triplets):".format(len(rfm)))
    
    raw_text_lens = []
    fixed_text_lens = []
    multi_text_lens = []
    
    for tid in rfm:
        raw_text_lens.append(len(raw[tid].get("modification", "").split()))
        fixed_text_lens.append(len(fixed[tid].get("modification", "").split()))
        multi_text_lens.append(len(multi[tid].get("modification", "").split()))
    
    import numpy as np
    for label, lens in [("Raw", raw_text_lens), ("Fixed", fixed_text_lens), ("Multi", multi_text_lens)]:
        print(f"  {label}: mean={np.mean(lens):.1f} words, median={np.median(lens):.1f}, "
              f"min={np.min(lens)}, max={np.max(lens)}")
    
    # Check for empty/missing modifications
    raw_empty = sum(1 for tid in rfm if not raw[tid].get("modification", "").strip())
    fixed_empty = sum(1 for tid in rfm if not fixed[tid].get("modification", "").strip())
    multi_empty = sum(1 for tid in rfm if not multi[tid].get("modification", "").strip())
    print(f"  Empty modifications - Raw: {raw_empty}, Fixed: {fixed_empty}, Multi: {multi_empty}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Strict intersection size: {len(rfm):,} triplets")
    print(f"  This is {100*len(rfm)/len(fixed):.1f}% of the Fixed/Multi 280K corpus")
    print(f"  Recommended training size: ~{len(rfm) - 20000:,} (after reserving 20K for dev+test)")
    print(f"  Recommended dev size: 10,000")
    print(f"  Recommended source test size: 10,000")
    
    # Save the strict intersection IDs
    out_dir = os.path.join(base, "data/pair_control")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "strict_intersection_ids.txt"), "w") as f:
        for tid in sorted(rfm):
            f.write(tid + "\n")
    
    print(f"\n  Strict intersection IDs saved to data/pair_control/strict_intersection_ids.txt")

if __name__ == "__main__":
    main()
