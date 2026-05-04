#!/usr/bin/env bash
# Run proposal datasets and write metrics under report/results/.
#
# This script can be expensive. SIFT1M and Deep1M each load up to 1M vectors.
# Deep1M also needs a 96-dimensional cluster, so the script requires
# RESET_STATE=1 before switching dimensions.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi

mkdir -p report/results

DURATION_SECONDS="${DURATION_SECONDS:-60}"
CONCURRENCY="${CONCURRENCY:-32}"
QUERY_LIMIT="${QUERY_LIMIT:-1000}"
RUN_DEEP="${RUN_DEEP:-0}"
RESET_STATE="${RESET_STATE:-0}"

wait_ready() {
  echo "Waiting for cluster readiness..."
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8100/health >/dev/null 2>&1 \
      && curl -fsS http://127.0.0.1:8200/ready >/dev/null 2>&1; then
      echo "Cluster is ready."
      return 0
    fi
    sleep 2
  done
  echo "Cluster did not become ready in time." >&2
  return 1
}

reset_state() {
  if [ "$RESET_STATE" != "1" ]; then
    echo "Refusing to clear shared_storage/ and wal/ without RESET_STATE=1." >&2
    echo "Set RESET_STATE=1 when you are okay deleting local runtime state." >&2
    exit 1
  fi
  docker compose down
  clear_runtime_state
}

maybe_reset_state() {
  if [ "$RESET_STATE" = "1" ]; then
    docker compose down
    clear_runtime_state
  fi
}

clear_runtime_state() {
  mkdir -p shared_storage wal
  echo "Clearing shared_storage/ and wal/ runtime state..."
  if rm -rf shared_storage/* wal/* 2>/dev/null; then
    return 0
  fi

  echo "Host cleanup hit permission-denied files; retrying cleanup via Docker root user..."
  docker compose run --rm --no-deps --user root writer-0 \
    sh -c 'rm -rf /shared_storage/* /wal/*'
}

start_cluster() {
  local dim="$1"
  local flush_threshold="$2"
  docker compose down
  VECSCALE_DIM="$dim" VECSCALE_SEGMENT_FLUSH_THRESHOLD="$flush_threshold" docker compose up -d --build
  wait_ready
}

echo "Starting 128D cluster for SIFT10K with a 10K flush threshold..."
maybe_reset_state
start_cluster 128 10000

echo "Running SIFT10K metrics..."
"$PYTHON_BIN" report/run_dataset_metrics.py \
  --dataset sift10k \
  --load \
  --qps \
  --recall \
  --duration "$DURATION_SECONDS" \
  --concurrency "$CONCURRENCY" \
  --query-limit 100 \
  --output report/results/sift10k_metrics.json

echo "Starting 128D cluster for SIFT1M with the normal 50K flush threshold..."
maybe_reset_state
start_cluster 128 50000

echo "Running SIFT1M metrics..."
"$PYTHON_BIN" report/run_dataset_metrics.py \
  --dataset sift1m \
  --load \
  --qps \
  --scaling \
  --auto-scale \
  --recall \
  --recall-queries 1000 \
  --recall-k 10 100 \
  --nprobe 32 \
  --duration "$DURATION_SECONDS" \
  --concurrency "$CONCURRENCY" \
  --query-limit "$QUERY_LIMIT" \
  --output report/results/sift1m_metrics.json

if [ "$RUN_DEEP" = "1" ]; then
  if [ ! -f data/deep1m/deep1m.hdf5 ]; then
    echo "Missing data/deep1m/deep1m.hdf5. Run bash report/download_deep1m.sh first." >&2
    exit 1
  fi

  echo "Switching to 96D cluster for Deep1M..."
  reset_state
  start_cluster 96 50000

  echo "Running Deep1M metrics..."
  "$PYTHON_BIN" report/run_dataset_metrics.py \
    --dataset deep1m \
    --load \
    --qps \
    --scaling \
    --auto-scale \
    --duration "$DURATION_SECONDS" \
    --concurrency "$CONCURRENCY" \
    --query-limit "$QUERY_LIMIT" \
    --output report/results/deep1m_metrics.json
fi

bash report/run_report.sh

echo "All requested dataset metrics complete."
