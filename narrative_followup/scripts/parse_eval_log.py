#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def main() -> None:
    log = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = log.read_text(errors="replace")
    matches = re.findall(r"^(Recall@1|Recall@5|Recall@10|Recall@50|mAP):\s+([0-9.]+)\s*$", text, re.M)
    if len(matches) < 5:
        raise SystemExit(f"incomplete metrics in {log}: {matches}")
    metrics = {k: float(v) for k, v in matches[-5:]}
    out.write_text(json.dumps({"source_log": str(log), "metrics": metrics}, indent=2) + "\n")


if __name__ == "__main__":
    main()
