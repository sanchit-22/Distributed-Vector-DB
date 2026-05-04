#!/usr/bin/env bash
# =============================================================================
# run_full_benchmark.sh — Run ALL VecScaleDB benchmarks for ALL datasets
# =============================================================================
#
# Datasets run:
#   1. SIFT10K  — 10K vectors, 128D, load + QPS + recall
#   2. SIFT1M   — 1M  vectors, 128D, load + QPS + recall + reader scaling
#   3. Deep1M   — 1M  vectors,  96D, load + QPS + recall + reader scaling
#                  (skipped if data file is missing; set RUN_DEEP=1 to require it)
#
# Usage:
#   bash report/run_full_benchmark.sh             # SIFT10K + SIFT1M only
#   RUN_DEEP=1 bash report/run_full_benchmark.sh  # include Deep1M
#
# Tunable env vars (all optional):
#   DURATION_SECONDS=60    QPS benchmark duration per dataset
#   CONCURRENCY=32         Concurrent search workers
#   NPROBE=32              IVF probe count (affects recall + QPS)
#   RECALL_QUERIES=1000    How many queries to use for recall evaluation
#
# Notes:
#   - The script ALWAYS clears shared_storage/ and wal/ between datasets.
#     Root-owned Docker files are removed via a Docker root-user helper so you
#     never hit "Permission denied".
#   - Docker images are rebuilt only for the first dataset; subsequent runs
#     reuse the built images unless --build is forced.
# =============================================================================
set -euo pipefail

# ── Locate project root ──────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Configuration ────────────────────────────────────────────────────────────
DURATION_SECONDS="${DURATION_SECONDS:-60}"
CONCURRENCY="${CONCURRENCY:-32}"
NPROBE="${NPROBE:-32}"
RECALL_QUERIES="${RECALL_QUERIES:-1000}"
RUN_DEEP="${RUN_DEEP:-0}"

PYTHON_BIN="python3"
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi

mkdir -p report/results

# ── Helper: print a section header ──────────────────────────────────────────
banner() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $*"
  echo "════════════════════════════════════════════════════════════════"
}

# ── Helper: clear shared_storage/ and wal/ (handles Docker root-owned files) ─
clear_state() {
  banner "Clearing runtime state (shared_storage/ and wal/)"
  mkdir -p shared_storage wal
  # Try host-side removal first (fast path)
  if rm -rf shared_storage/* wal/* 2>/dev/null; then
    echo "✓ Cleared by host user."
    return 0
  fi
  # Fall back to Docker root-user removal (handles Docker-created root files)
  echo "Host rm hit permission-denied files — using Docker root-user fallback..."
  docker compose run --rm --no-deps --user root writer-0 \
    sh -c 'rm -rf /shared_storage/* /wal/* && echo "✓ Cleared via Docker root."'
}

# ── Helper: wait until writer + at least one reader are healthy ──────────────
wait_ready() {
  echo "Waiting for cluster to become ready..."
  local max_attempts=60
  for i in $(seq 1 "$max_attempts"); do
    if curl -fsS http://127.0.0.1:8100/health >/dev/null 2>&1 \
      && curl -fsS http://127.0.0.1:8200/ready >/dev/null 2>&1; then
      echo "✓ Cluster ready (attempt ${i})."
      return 0
    fi
    echo "  attempt ${i}/${max_attempts} — not ready yet, sleeping 2s..."
    sleep 2
  done
  echo "✗ Cluster did not become ready after $((max_attempts * 2))s." >&2
  docker compose logs --tail=40 >&2
  exit 1
}

# ── Helper: stop, clear, and start cluster for a given DIM + flush threshold ─
restart_cluster() {
  local dim="$1"
  local flush_threshold="$2"
  local rebuild="${3:-0}"   # pass "1" to force --build

  banner "Restarting cluster  DIM=${dim}  flush_threshold=${flush_threshold}"
  docker compose down

  clear_state

  local build_flag=""
  if [ "$rebuild" = "1" ]; then
    build_flag="--build"
  fi

  VECSCALE_DIM="$dim" \
  VECSCALE_SEGMENT_FLUSH_THRESHOLD="$flush_threshold" \
  docker compose up -d $build_flag

  wait_ready
}

# ── Helper: run the Python benchmark for one dataset ────────────────────────
run_metrics() {
  local dataset="$1"       # sift10k | sift1m | deep1m
  local query_limit="$2"   # how many test queries to use
  local output="$3"        # output JSON path
  shift 3
  local extra_flags=("$@") # any additional flags (--scaling --auto-scale etc.)

  banner "Running metrics for ${dataset^^}"
  "$PYTHON_BIN" report/run_dataset_metrics.py \
    --dataset      "$dataset"      \
    --load                         \
    --qps                          \
    --recall                       \
    --recall-queries "$RECALL_QUERIES" \
    --recall-k 10 100              \
    --nprobe       "$NPROBE"       \
    --duration     "$DURATION_SECONDS" \
    --concurrency  "$CONCURRENCY"  \
    --query-limit  "$query_limit"  \
    --output       "$output"       \
    "${extra_flags[@]}"

  echo "✓ Results written to $output"
}

# =============================================================================
# DATASET 1 — SIFT10K  (128D, flush every 10K, no scaling sweep)
# =============================================================================
if [ ! -f data/sift1m/sift1m.hdf5 ]; then
  echo "✗ Missing data/sift1m/sift1m.hdf5 (used for SIFT10K and SIFT1M)." >&2
  echo "  Download it from http://ann-benchmarks.com/sift-128-euclidean.hdf5" >&2
  echo "  and place it at data/sift1m/sift1m.hdf5" >&2
  exit 1
fi

restart_cluster 128 10000 1   # 1 = force --build on first start

run_metrics sift10k 100 \
  report/results/sift10k_metrics.json

# =============================================================================
# DATASET 2 — SIFT1M  (128D, flush every 50K, includes reader scaling sweep)
# =============================================================================
restart_cluster 128 50000

run_metrics sift1m 1000 \
  report/results/sift1m_metrics.json \
  --scaling --auto-scale

# =============================================================================
# DATASET 3 — Deep1M  (96D, flush every 50K, includes reader scaling sweep)
#             Skipped unless data file exists and RUN_DEEP=1
# =============================================================================
if [ "$RUN_DEEP" = "1" ]; then
  if [ ! -f data/deep1m/deep1m.hdf5 ]; then
    echo "✗ RUN_DEEP=1 but data/deep1m/deep1m.hdf5 is missing." >&2
    echo "  Run:  bash report/download_deep1m.sh" >&2
    exit 1
  fi

  restart_cluster 96 50000   # DIM switch requires fresh state

  run_metrics deep1m 1000 \
    report/results/deep1m_metrics.json \
    --scaling --auto-scale
else
  banner "Skipping Deep1M (set RUN_DEEP=1 to include it)"
fi

# =============================================================================
# Final: regenerate the LaTeX PDF
# =============================================================================
banner "Regenerating report PDF"
bash report/run_report.sh

banner "ALL DONE"
echo ""
echo "Results:"
echo "  report/results/sift10k_metrics.json"
echo "  report/results/sift1m_metrics.json"
if [ "$RUN_DEEP" = "1" ]; then
  echo "  report/results/deep1m_metrics.json"
fi
echo "  report/latex/vecscaledb_report.pdf"
echo ""
