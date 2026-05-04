#!/usr/bin/env bash
# Regenerate the benchmark report PDF from local metrics.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi

"$PYTHON_BIN" report/collect_metrics.py
"$PYTHON_BIN" report/generate_report.py

echo "Report written to report/vecscaledb_ann_benchmark_report.pdf"
