# VecScaleDB Phase 8 Guide

This document explains:

- what Phase 8 adds,
- how structured logs and metrics work,
- how to run observability checks,
- how to complete the final benchmark report.

Scope: current repository state under `Distributed-Vector-DB`.

## 1. High-Level Architecture

Phase 8 adds observability and final evaluation support.

Implemented features:

- Shared observability helper.
- JSON structured request logs.
- Per-request `trace_id`.
- Trace propagation via `x-trace-id`.
- Per-service `/metrics` endpoint.
- Request counters.
- Request duration summaries.
- Search duration summaries.
- Service-specific gauges.
- Benchmark report file.

## 2. Observability Design

All FastAPI services use:

```python
from vecscaledb.observability import attach_observability, configure_logging
```

Each app:

1. Creates `Settings`.
2. Configures JSON logging.
3. Creates the FastAPI app.
4. Calls `attach_observability(...)`.

The middleware logs:

- `request.start`
- `request.end`
- `request.error` if an exception occurs

Every request log includes:

- `node_role`
- `node_id`
- `trace_id`
- HTTP method
- path
- status code on completion
- duration in milliseconds

## 3. Trace IDs

If a client sends:

```text
x-trace-id: my-trace-id
```

the same value is:

- attached to request logs,
- returned as response header `x-trace-id`.

If the client does not send a trace ID, the service generates one.

## 4. Metrics Endpoint

Every service exposes:

```text
GET /metrics
```

Examples:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
Invoke-RestMethod http://127.0.0.1:8100/metrics
Invoke-RestMethod http://127.0.0.1:8200/metrics
```

Metrics format is Prometheus-style text.

Core metrics:

- `vecscaledb_request_total`
- `vecscaledb_request_duration_seconds_count`
- `vecscaledb_request_duration_seconds_sum`
- `vecscaledb_search_duration_seconds_count`
- `vecscaledb_search_duration_seconds_sum`

Example labels:

- `node_role`
- `node_id`
- `method`
- `path`
- `status`

## 5. Service Gauges

### Coordinator

- `vecscaledb_ring_size`
- `vecscaledb_registered_nodes`
- `vecscaledb_registered_readers`
- `vecscaledb_coordinator_leader`

### Writer

- `vecscaledb_wal_lsn`
- `vecscaledb_segments_loaded`
- `vecscaledb_memtable_size`
- `vecscaledb_active_snapshots`

### Reader

- `vecscaledb_segments_loaded`
- `vecscaledb_vectors_loaded`
- `vecscaledb_reader_ready`

### Single Node

- `vecscaledb_wal_lsn`
- `vecscaledb_segments_loaded`
- `vecscaledb_memtable_size`
- `vecscaledb_active_snapshots`

## 6. File-by-File Reference

- `vecscaledb/observability.py`
  - Shared logging and metrics helper.
  - Defines `MetricsRegistry`.
  - Defines `configure_logging`.
  - Defines `attach_observability`.

- `vecscaledb/node.py`
  - Single-node app with observability attached.

- `vecscaledb/coordinator/app.py`
  - Coordinator app with observability and coordinator gauges.

- `vecscaledb/writer/app.py`
  - Writer app with observability and WAL/storage gauges.

- `vecscaledb/reader/app.py`
  - Reader app with observability and cache/readiness gauges.

- `benchmark_report.md`
  - Final benchmark report file.

- `tests/unit/test_observability.py`
  - Verifies trace headers, metrics, and JSON request logs.

## 7. How To Run

Run from project root:

```powershell
cd "C:\Users\priya\OneDrive\Desktop\SEM 4\Distri\distriProject\Distributed-Vector-DB"
```

Start Docker stack:

```powershell
docker compose up -d --build
```

Send traced search:

```powershell
$query = @(0.0, 1.0) + @(0.0) * 126
$body = @{
  query = $query
  top_k = 5
  nprobe = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Headers @{ "x-trace-id" = "manual-trace-1" } `
  -Body $body `
  -ContentType "application/json"
```

Check metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
Invoke-RestMethod http://127.0.0.1:8100/metrics
Invoke-RestMethod http://127.0.0.1:8200/metrics
```

## 8. How To Test

### Observability Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_observability.py -v
```

### Full Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 9. Benchmark Report

The report lives at:

```text
benchmark_report.md
```

Current status:

- Live smoke numbers are included.
- Full SIFT1M numbers are marked pending because the dataset file is not present locally.

To complete final SIFT1M results:

1. Download SIFT1M:

```powershell
bash data/download_sift.sh
```

2. Load the dataset:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.load_dataset --dataset data\sift1m\sift1m.hdf5 --writer-url http://127.0.0.1:8100 --batch-size 10000
```

3. Run QPS benchmark:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.benchmark_qps --dataset data\sift1m\sift1m.hdf5 --coordinator-url http://127.0.0.1:8000 --concurrency 32 --duration 60
```

4. Update `benchmark_report.md`.

## 10. Known Boundaries

- Metrics are lightweight in-process text metrics, not a full Prometheus server.
- Metrics reset when a process restarts.
- Full benchmark numbers require the external SIFT1M HDF5 file.
- Phase 8 does not add dashboards; it exposes enough logs/metrics for external tooling.
