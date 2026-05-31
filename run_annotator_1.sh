#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python app.py --annotator Annotator_1 --server-port 7860
