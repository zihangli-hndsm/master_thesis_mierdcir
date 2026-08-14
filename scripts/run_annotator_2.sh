#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/app.py --annotator Annotator_2 --server-port 7861
