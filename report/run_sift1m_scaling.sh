#!/usr/bin/env bash
# Run the SIFT1M QPS-vs-reader-count benchmark and store results under report/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p report/results

PYTHON_BIN="${PYTHON:-python3}"
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi

"$PYTHON_BIN" -m vecscaledb.bench.scaling_eval \
  --dataset data/sift1m/sift1m.hdf5 \
  --coordinator-url http://127.0.0.1:8000 \
  --reader-counts 1 2 3 4 5 \
  --duration "${DURATION_SECONDS:-60}" \
  --concurrency "${CONCURRENCY:-32}" \
  --query-limit "${QUERY_LIMIT:-1000}" \
  --output report/results/sift1m_scaling_results.json \
  --auto-scale

echo "Scaling results written to report/results/sift1m_scaling_results.json"
