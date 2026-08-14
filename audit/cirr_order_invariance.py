#!/usr/bin/env python3
"""CIRR query-order invariance (plan §5): re-export with a shuffled DataLoader
and verify the pairid-aligned prediction JSON is byte-identical to the
order-preserving export.
"""
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from eval_checkpoint import EvalJsonDataset, evaluate_cirr_and_dump

CKPT = "checkpoints/topk_pair_multi_s42/topk_epoch_0002_step_000600_score_0.441448.pth.tar"
METHOD = "cross_attn_alpha"
REF_JSON = "cirr_test_results/val_rehearsal/val_rehearsal_recall_bs128.json"


def main():
    device = torch.device("cuda")
    from models.full_model import ScheiCIR
    model = ScheiCIR(method=METHOD, temperature=0.07).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    ds = EvalJsonDataset(
        data_path="./data/CIRR",
        json_path="cap.rc2.val.json",
        lmdb_path="./data/CIRR/images_224_lmdb",
        require_target=False,
        split_json_path=None,
    )
    loader_shuffled = DataLoader(
        ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=EvalJsonDataset.collate_fn,
    )
    out = evaluate_cirr_and_dump(model, loader_shuffled, "order_inv", device,
                                 dataset_version="rc2", metric="recall")
    shuffled_path = "cirr_test_results/val_rehearsal/val_rehearsal_recall_shuffled.json"
    with open(shuffled_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    ref = json.load(open(REF_JSON))
    mismatches = []
    for key in ref:
        if key in ("version", "metric"):
            continue
        if out.get(key) != ref[key]:
            mismatches.append(key)
    same = len(mismatches) == 0
    print(f"pairid-aligned mismatch count: {len(mismatches)} / {len(ref) - 2}")
    print(f"QUERY-ORDER INVARIANCE: {'PASS' if same else 'FAIL'}")
    if not same:
        print("first mismatches:", mismatches[:5])
    with open("audit/cirr_order_invariance.json", "w") as f:
        json.dump({"pass": same, "mismatches": len(mismatches), "n_queries": len(ref) - 2}, f, indent=2)


if __name__ == "__main__":
    main()
