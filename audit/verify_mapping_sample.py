#!/usr/bin/env python3
"""Manual-check support for plan §7.4: verify the 100-query mapping sample.
Checks (all automated): reference/target exist in LMDB, ref != target,
gallery index consistent with evaluate()'s unique-target gallery construction.
Prints a table of 100 rows (query id, ref, target, gallery idx) for spot-checking.
"""
import json
import lmdb
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMDB = f"{BASE}/data/MTCIR/images_224_lmdb"
TEST = f"{BASE}/data/pair_control/test.jsonl"
SAMPLE = f"{BASE}/audit/mapping_sample_100.json"

def key_candidates(image_id):
    image_id = str(image_id).strip()
    normalized = image_id.replace("\\", "/").lstrip("./")
    base, ext = os.path.splitext(normalized)
    cands = [normalized]
    if not ext:
        cands += [f"{normalized}.png", f"{normalized}.jpg", f"{normalized}.jpeg"]
    return list(dict.fromkeys(cands))

def main():
    recs = [json.loads(l) for l in open(TEST) if l.strip()]
    by_id = {r["id"]: r for r in recs}
    targets = [r["target_img"] for r in recs]
    gallery = list(dict.fromkeys(targets))
    g_index = {img: i for i, img in enumerate(gallery)}

    sample = json.load(open(SAMPLE))
    print(f"sample size: {len(sample)}")

    env = lmdb.open(LMDB, readonly=True, lock=False, readahead=False, meminit=False)
    fails = []
    with env.begin(buffers=True) as txn:
        for row in sample:
            qid = row["query_id"]
            rec = by_id.get(qid)
            issues = []
            if rec is None:
                issues.append("record missing in test.jsonl")
            else:
                if rec["image"] != row["reference"]:
                    issues.append(f"reference mismatch: {rec['image']} != {row['reference']}")
                if rec["target_img"] != row["target"]:
                    issues.append(f"target mismatch: {rec['target_img']} != {row['target']}")
                if row["gallery_index"] != g_index.get(rec["target_img"]):
                    issues.append(f"gallery idx mismatch: {row['gallery_index']} != {g_index.get(rec['target_img'])}")
            if row["reference"] == row["target"]:
                issues.append("ref == target")
            if not any(txn.get(c.encode()) for c in key_candidates(row["reference"])):
                issues.append("reference missing in LMDB")
            if not any(txn.get(c.encode()) for c in key_candidates(row["target"])):
                issues.append("target missing in LMDB")
            if issues:
                fails.append((qid, issues))

    env.close()
    print(f"rows with issues: {len(fails)} / {len(sample)}")
    for qid, issues in fails[:10]:
        print(f"  {qid}: {issues}")

    # dump a printable spot-check table
    with open(f"{BASE}/audit/mapping_spotcheck_table.md", "w") as f:
        f.write("| # | query_id | reference | target | gallery_idx |\n|---|---|---|---|---|\n")
        for i, row in enumerate(sample, 1):
            f.write(f"| {i} | {row['query_id']} | {row['reference']} | {row['target']} | {row['gallery_index']} |\n")
    print(f"Wrote audit/mapping_spotcheck_table.md ({len(sample)} rows)")
    if fails:
        print(f"RESULT: {len(fails)} issues found - review audit/mapping_spotcheck_table.md")
    else:
        print("RESULT: all 100 sample rows consistent (ref/target/gallery-index/LMDB)")


if __name__ == "__main__":
    main()
