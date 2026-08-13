#!/usr/bin/env python3
"""MTCIR data-level audit (plan §7.3/§7.4/§7.5):
  1. Split & image leakage checks (triplet / reference / target / any-role overlap)
  2. Gallery & positive mapping checks (query count, gallery ids, gold column indices, 100-query sample)
  3. Metric unit tests on synthetic embeddings (6 properties from §7.5)
Outputs: audit/mtcir_data_audit.json + printed summary.
"""
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__)) + "/.."
SPLITS = {
    "train": f"{BASE}/data/pair_control/train.jsonl",
    "dev": f"{BASE}/data/pair_control/dev.jsonl",
    "test": f"{BASE}/data/pair_control/test.jsonl",
}
OUT = {}


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def roles(recs):
    refs = set()
    tgts = set()
    for r in recs:
        refs.add(r.get("image", ""))
        tgts.add(r.get("target_img", ""))
    return refs, tgts


def overlap(a, b):
    return len(a & b)


# ---------------------------------------------------------------- §7.3 leakage
def check_leakage():
    print("=" * 70)
    print("[§7.3] SPLIT / IMAGE LEAKAGE CHECKS")
    print("=" * 70)
    data = {}
    for name, path in SPLITS.items():
        recs = load_jsonl(path)
        refs, tgts = roles(recs)
        data[name] = {
            "n": len(recs),
            "ids": {r["id"] for r in recs},
            "refs": refs,
            "tgts": tgts,
            "all_imgs": refs | tgts,
        }
        print(f"  {name}: {len(recs)} triplets, {len(refs)} unique refs, {len(tgts)} unique targets")

    def pair_stat(a, b):
        ia, ib = data[a], data[b]
        return {
            "triplet_id_overlap": overlap(ia["ids"], ib["ids"]),
            "ref_overlap": overlap(ia["refs"], ib["refs"]),
            "target_overlap": overlap(ia["tgts"], ib["tgts"]),
            "any_role_overlap": overlap(ia["all_imgs"], ib["all_imgs"]),
            # directed leaks
            f"{a}_target_into_{b}_reference": overlap(ia["tgts"], ib["refs"]),
            f"{a}_reference_into_{b}_target": overlap(ia["refs"], ib["tgts"]),
        }

    res = {
        "sizes": {k: data[k]["n"] for k in data},
        "unique_refs": {k: len(data[k]["refs"]) for k in data},
        "unique_targets": {k: len(data[k]["tgts"]) for k in data},
        "train_dev": pair_stat("train", "dev"),
        "train_test": pair_stat("train", "test"),
        "dev_test": pair_stat("dev", "test"),
        "three_way_triplet_intersection": len(
            data["train"]["ids"] & data["dev"]["ids"] & data["test"]["ids"]
        ),
    }
    for pair in ("train_dev", "train_test", "dev_test"):
        print(f"  {pair}: {json.dumps(res[pair])}")
    print(f"  three-way triplet intersection: {res['three_way_triplet_intersection']}")
    OUT["leakage"] = res


# ---------------------------------------------------------------- §7.4 mapping
def check_mapping():
    print()
    print("=" * 70)
    print("[§7.4] GALLERY & POSITIVE MAPPING (pair_control/test.jsonl)")
    print("=" * 70)
    recs = load_jsonl(SPLITS["test"])
    n = len(recs)
    targets = [r["target_img"] for r in recs]
    refs = [r["image"] for r in recs]
    gallery = list(dict.fromkeys(targets))  # same construction as evaluate()'s chroma gallery
    g_index = {img: i for i, img in enumerate(gallery)}

    missing = [r["id"] for r, t in zip(recs, targets) if t not in g_index]
    ref_is_target = sum(1 for r, t in zip(refs, targets) if r == t)
    dup_targets = len(targets) - len(gallery)
    target_counts = Counter(targets)
    multi_gold = {t: c for t, c in target_counts.items() if c > 1}

    gold_cols = [g_index[t] for t in targets]
    col_hist = Counter(gold_cols)
    # rank distribution when ranked by gallery insertion order is meaningless; report column histogram tail
    top_cols = sorted(col_hist.items())[:10]

    rng = random.Random(42)
    sample = rng.sample(list(range(n)), min(100, n))
    sample_rows = [
        {"query_id": recs[i]["id"], "reference": refs[i], "target": targets[i],
         "gallery_index": gold_cols[i]}
        for i in sample
    ]

    res = {
        "n_queries": n,
        "gallery_unique_ids": len(gallery),
        "targets_missing_from_gallery": len(missing),
        "ref_equals_target": ref_is_target,
        "duplicate_target_ids": dup_targets,
        "targets_with_multiple_queries": len(multi_gold),
        "gold_column_histogram_head": top_cols,
        "sample_100_queries": sample_rows,
    }
    print(f"  queries: {n}; gallery unique ids: {len(gallery)}")
    print(f"  targets missing from gallery: {len(missing)} (must be 0)")
    print(f"  ref == target: {ref_is_target}; duplicate target ids: {dup_targets}")
    print(f"  targets shared by >1 query: {len(multi_gold)}")
    print(f"  gold column histogram (first 10): {top_cols}")
    OUT["mapping"] = res
    with open(f"{BASE}/audit/mapping_sample_100.json", "w") as f:
        json.dump(sample_rows, f, indent=2)
    print(f"  100-query sample written to audit/mapping_sample_100.json")


# ---------------------------------------------------------------- §7.5 metric unit tests
def rank_metrics(query_embs, gallery_embs, gold_indices, k_list=(1, 5, 10), chunk=128):
    """Pure-numpy global-gallery ranking metric (mirrors eval logic)."""
    recalls = {k: [] for k in k_list}
    aps = []
    n_q = len(query_embs)
    for start in range(0, n_q, chunk):
        q = query_embs[start:start + chunk]
        scores = gallery_embs @ q.T  # (G, chunk)
        for j in range(q.shape[0]):
            g = gold_indices[start + j]
            order = np.argsort(-scores[:, j])
            rank = int(np.where(order == g)[0][0]) + 1
            aps.append(1.0 / rank)
            for k in k_list:
                recalls[k].append(1 if rank <= k else 0)
    return {f"R@{k}": float(np.mean(recalls[k])) for k in k_list}, float(np.mean(aps))


def metric_unit_tests():
    print()
    print("=" * 70)
    print("[§7.5] METRIC UNIT TESTS (synthetic embeddings)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    N, D = 1000, 64
    gallery = rng.normal(size=(N, D))
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    results = {}

    # T1: query == target -> R@1 = 100%
    q1 = gallery.copy()
    results["T1_identity_R1_100pct"] = rank_metrics(q1, gallery, np.arange(N))[0]["R@1"]

    # T2: random queries -> R@1 ~ 1/N
    q2 = rng.normal(size=(N, D))
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    r2 = rank_metrics(q2, gallery, np.arange(N))[0]
    results["T2_random_R1"] = r2["R@1"]
    results["T2_random_expected"] = 1.0 / N

    # T3: permuted gold indices -> near-random
    perm = rng.permutation(N)
    r3 = rank_metrics(q2, gallery, perm)[0]
    results["T3_permuted_gold_R1"] = r3["R@1"]

    # T4: shuffled query order -> identical metrics
    order = rng.permutation(N)
    r4a = rank_metrics(q1, gallery, np.arange(N))
    r4b = rank_metrics(q1[order], gallery, np.arange(N)[order])
    results["T4_shuffle_order_max_diff"] = max(
        abs(r4a[0][k] - r4b[0][k]) for k in r4a[0]
    ) + abs(r4a[1] - r4b[1])

    # T5: target forced to rank 2 -> R@1=0, R@5=100%
    # explicit construction: gold has sim 0.5 (2nd), decoy 0.9 (1st), noise ~0
    e0 = np.zeros(D); e0[0] = 1.0
    gold = np.zeros(D); gold[0] = 0.5; gold[1] = np.sqrt(1 - 0.25)
    decoy = np.zeros(D); decoy[0] = 0.9; decoy[1] = np.sqrt(1 - 0.81)
    noise = rng.normal(size=(N - 2, D))
    noise -= noise @ e0[:, None] * e0[None, :]  # orthogonal to e0
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    g5 = np.vstack([gold, decoy, noise])
    q5 = e0.reshape(1, -1)
    r5 = rank_metrics(q5, g5, np.array([0]), k_list=(1, 5, 10))[0]
    results["T5_target_rank2_R1"] = r5["R@1"]
    results["T5_target_rank2_R5"] = r5["R@5"]

    # T6: chunked vs full-batch ranking identical
    r6a = rank_metrics(q1, gallery, np.arange(N), chunk=N)
    r6b = rank_metrics(q1, gallery, np.arange(N), chunk=7)
    results["T6_chunk_invariance_max_diff"] = max(
        abs(r6a[0][k] - r6b[0][k]) for k in r6a[0]
    ) + abs(r6a[1] - r6b[1])

    for k, v in results.items():
        print(f"  {k}: {v}")
    OUT["metric_unit_tests"] = results


def main():
    check_leakage()
    check_mapping()
    metric_unit_tests()
    with open(f"{BASE}/audit/mtcir_data_audit.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print()
    print("Wrote audit/mtcir_data_audit.json")


if __name__ == "__main__":
    main()
