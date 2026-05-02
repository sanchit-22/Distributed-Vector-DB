# VecScaleDB Benchmark Report

## Dataset Status

SIFT1M file expected at:

```text
data/sift1m/sift1m.hdf5
```

Current local status: dataset file not present during Phase 8 implementation.

The system has been verified with the currently loaded Docker dataset:

- vectors: 210,003
- persisted segments: 7
- readers: 5
- reader mode: full replication
- coordinator fanout: replicated round-robin

## Recall Evaluation

| Config | Recall@10 | Recall@100 | Status |
|---|---:|---:|---|
| Single node SIFT10K | Not rerun in Phase 8 | Not rerun in Phase 8 | Available via `vecscaledb.bench.recall_eval` |
| 5-reader cluster SIFT1M | Pending | Pending | Requires `data/sift1m/sift1m.hdf5` |

Run after downloading SIFT1M:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.recall_eval --host 127.0.0.1 --port 8000 --dataset data\sift1m\sift1m.hdf5
```

## QPS Scaling

| Readers | QPS | p50 ms | p95 ms | p99 ms | Scaling efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | Pending | Pending | Pending | Pending | 1.00x |
| 2 | Pending | Pending | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending | Pending | Pending |
| 5 | 108.07 | 69.63 | 84.70 | 91.89 | Smoke only |

Earlier Phase 6/7 live QPS smoke result on a 160,001-vector dataset:

```text
total_queries=328
errors=0
elapsed=3.04s
qps=108.07
p50=69.63ms
p95=84.70ms
p99=91.89ms
```

Latest end-to-end validation smoke:

```text
date=2026-05-03
automated_tests=66 passed
docker_services=10 up
reader_health=5 healthy
fresh_insert_ids=990001,990002
fresh_insert_lsn=215
fresh_search_top_id=990001
fresh_search_top_distance=0.0
fresh_search_incomplete=False
bad_writer_insert_status=400
bad_reader_search_status=400
bad_coordinator_search_status=422
reader_stop_searches=10
reader_stop_http_5xx=0
reader_stop_found_990001=10
coordinator_0_stopped_search_via_8001=passed
```

Run final 60-second SIFT1M benchmark:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.benchmark_qps --dataset data\sift1m\sift1m.hdf5 --coordinator-url http://127.0.0.1:8000 --concurrency 32 --duration 60
```

## Fault Tolerance

| Scenario | Recovery time | Data loss | Status |
|---|---:|---|---|
| Reader crash + restart | Under 30 s in Docker smoke | None observed | Passed |
| Writer crash + restart | Under 10 s in Docker smoke | None observed | Passed |
| Coordinator leader failover | About 15 s in Docker smoke | None observed | Passed |

## Observability

Implemented in Phase 8:

- JSON structured request logs.
- `trace_id` propagation through `x-trace-id`.
- `node_role` and `node_id` on request logs.
- Per-service `/metrics` endpoint.
- Request counters and duration summaries.
- Search duration summary.
- Service gauges such as WAL LSN, loaded segments, reader readiness, and coordinator ring size.

Metrics examples:

```text
http://127.0.0.1:8000/metrics
http://127.0.0.1:8100/metrics
http://127.0.0.1:8200/metrics
```

## Final SIFT1M Checklist

- [ ] Download `data/sift1m/sift1m.hdf5`.
- [ ] Load all 1,000,000 vectors.
- [ ] Confirm coordinator `/segments` shows expected persisted segments.
- [ ] Run Recall@10 / Recall@100.
- [ ] Run 1-reader QPS benchmark.
- [ ] Run 2-reader QPS benchmark.
- [ ] Run 3-reader QPS benchmark.
- [ ] Run 5-reader QPS benchmark.
- [ ] Fill final QPS scaling table with real SIFT1M numbers.
