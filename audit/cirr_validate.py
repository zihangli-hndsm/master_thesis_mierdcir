#!/usr/bin/env python3
"""CIRR test1 submission validator (plan §4).

Checks (each reported PASS/FAIL):
 1. JSON parses with standard parser
 2. version == "rc2"
 3. metric == "recall" | "recall_subset"
 4. every pairid in annotations appears exactly once; no extra/missing keys
 5. recall: exactly 50 unique candidates/query; recall_subset: exactly 3 unique
 6. all predicted image ids exist in the test1 image split
 7. reference image is not in the prediction list
 8. subset predictions are within the query's img_set.members
 9. file size < 5 MB
10. records SHA256 of prediction, checkpoint (optional) and generating command
Usage:
  python cirr_validate.py --pred cirr_test_results/raw_s42_recall.json \
      --gt data/CIRR/cap.rc2.test1.json \
      --split data/CIRR/split.rc2.test1.json \
      [--checkpoint path.pth.tar]
"""
import argparse
import hashlib
import json
import os
import sys

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True, help="cap.rc2.test1.json")
    ap.add_argument("--split", required=True, help="split.rc2.test1.json")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    checks = []
    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    print(f"Validating {args.pred}")
    # 1. parse + 9. size
    size = os.path.getsize(args.pred)
    check("file_size_lt_5MB", size < 5 * 1024 * 1024, f"{size/1e6:.2f} MB")
    with open(args.pred, "r", encoding="utf-8") as f:
        pred = json.load(f)

    # 2-3. version/metric
    check("version_rc2", pred.get("version") == "rc2", str(pred.get("version")))
    metric = pred.get("metric")
    check("metric_valid", metric in ("recall", "recall_subset"), str(metric))

    gt = json.load(open(args.gt))
    gt_by_id = {r["pairid"]: r for r in gt}
    split = json.load(open(args.split))
    split_ids = set(split.keys())

    pred_keys = [k for k in pred if k not in ("version", "metric")]
    # 4. key coverage
    missing = [str(q) for q in gt_by_id if str(q) not in pred]
    extra = [k for k in pred_keys if int(k) not in gt_by_id]
    check("keys_no_missing", not missing, f"missing={len(missing)}")
    check("keys_no_extra", not extra, f"extra={len(extra)}")
    dup = len(pred_keys) != len(set(pred_keys))
    check("keys_unique", not dup)

    expected = 50 if metric == "recall" else 3
    bad_len, bad_dup, bad_split, bad_ref, bad_subset = [], [], [], [], []
    for k in pred_keys:
        cands = pred[k]
        if not isinstance(cands, list) or len(cands) != expected:
            bad_len.append(k)
        if len(set(cands)) != len(cands):
            bad_dup.append(k)
        for c in cands:
            if c not in split_ids:
                bad_split.append((k, c))
        rec = gt_by_id[int(k)]
        ref = rec.get("reference")
        if ref in cands:
            bad_ref.append(k)
        if metric == "recall_subset":
            members = set(rec.get("img_set", {}).get("members", []))
            for c in cands:
                if c not in members:
                    bad_subset.append((k, c))

    check(f"candidates_len_{expected}", not bad_len, f"bad={len(bad_len)}")
    check("candidates_unique", not bad_dup, f"bad={len(bad_dup)}")
    check("candidates_in_split", not bad_split, f"bad={len(bad_split)}")
    check("no_reference_in_preds", not bad_ref, f"bad={len(bad_ref)}")
    if metric == "recall_subset":
        check("subset_within_members", not bad_subset, f"bad={len(bad_subset)}")

    # 10. hashes + command
    manifest = {
        "prediction_file": os.path.abspath(args.pred),
        "prediction_sha256": sha256(args.pred),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "generating_command": " ".join(sys.argv),
        "all_checks_passed": all(ok for _, ok, _ in checks),
    }
    mpath = args.pred + ".validation.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written to {mpath}")
    print(f"\nOVERALL: {'PASS' if manifest['all_checks_passed'] else 'FAIL'}")
    sys.exit(0 if manifest["all_checks_passed"] else 1)

if __name__ == "__main__":
    main()
