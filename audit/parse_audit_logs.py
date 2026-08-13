#!/usr/bin/env python3
"""Parse audit_mtcir_*.log into a table (tag -> metrics)."""
import json
import re
import sys

TAG_RE = re.compile(r"^=== (\S+) ===")
M_RE = re.compile(r"^(Recall@\d+|mAP):\s*([-+0-9.eE]+)\s*$")

def parse(path):
    out = {}
    tag = None
    metrics = {}
    for line in open(path):
        m = TAG_RE.match(line.strip())
        if m:
            if tag:
                out[tag] = metrics
            tag, metrics = m.group(1), {}
            continue
        m = M_RE.match(line.strip())
        if m and tag:
            metrics[m.group(1)] = float(m.group(2))
    if tag:
        out[tag] = metrics
    return out

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    import glob, os
    if not path:
        logs = sorted(glob.glob("logs/audit_mtcir_*.log"), key=os.path.getmtime)
        path = logs[-1]
    table = parse(path)
    print(f"# {path}\n")
    print("| tag | R@1 | R@5 | R@10 | R@50 | mAP |")
    print("|---|---|---|---|---|---|")
    for tag, m in table.items():
        print(f"| {tag} | {m.get('Recall@1','-')} | {m.get('Recall@5','-')} | "
              f"{m.get('Recall@10','-')} | {m.get('Recall@50','-')} | {m.get('mAP','-')} |")
    with open("audit/mtcir_gpu_audit_table.json", "w") as f:
        json.dump(table, f, indent=2)
    print("\nWrote audit/mtcir_gpu_audit_table.json")
