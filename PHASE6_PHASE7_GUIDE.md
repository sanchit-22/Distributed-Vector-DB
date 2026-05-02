# VecScaleDB Phase 6/7 Guide

This document explains:

- what Phase 6 and Phase 7 add,
- how horizontal reader scaling works,
- how fault tolerance is handled,
- how to run and test the implementation.

Scope: current repository state under `Distributed-Vector-DB`.

## 1. High-Level Architecture

### Phase 6: Horizontal Reader Scaling

Phase 6 turns the single-reader search path into a five-reader benchmark topology.

Implemented features:

- `reader-1` through `reader-4` are enabled in Docker Compose.
- Readers run in full-replication mode with `VECSCALE_READER_LOAD_ALL_SHARDS=true`.
- Coordinator supports `replicated` search fanout mode.
- In replicated mode, each query is sent to one reader in round-robin order.
- SIFT HDF5 streaming loader is available.
- QPS benchmark script is available.

### Phase 7: Fault Tolerance

Phase 7 hardens the cluster against node crashes and coordinator failover.

Implemented features:

- Reader `/ready` endpoint.
- Docker healthchecks for reader containers.
- Writer and readers use all coordinator URLs.
- Coordinator leader TTL is refreshed.
- Coordinator stale local leader state is cleared.
- Reader/writer background loops survive transient coordinator failures.
- Writer WAL recovery is tested for unflushed acknowledged writes.
- Live failover checks were verified with Docker.

## 2. Phase 6 Reader Scaling Flow

### Full Replication Mode

1. Writer creates shard-0 segments.
2. Coordinator stores segment metadata.
3. All readers load all segment metadata because `reader_load_all_shards=true`.
4. Coordinator receives a client search.
5. Coordinator chooses one reader in round-robin order.
6. Chosen reader searches its local full segment cache.
7. Coordinator returns the reader result.

This avoids duplicate work across five readers and allows QPS to scale with reader count.

### Benchmark Flow

1. Download SIFT1M.
2. Load vectors into writer using `vecscaledb.bench.load_dataset`.
3. Wait for writer flushes and reader refresh.
4. Run `vecscaledb.bench.benchmark_qps`.
5. Compare 1-reader and 5-reader QPS.

## 3. Phase 7 Fault-Tolerance Flow

### Reader Restart

1. Stop a reader container.
2. Coordinator refreshes membership and continues using available readers.
3. Restart the reader.
4. Reader registers again with coordinator.
5. Reader refreshes its segment cache.
6. `/ready` returns 200 after the first refresh.

### Writer Crash Recovery

1. Writer acknowledges inserts only after WAL append/fsync.
2. If writer crashes before flush, WAL records remain.
3. On restart, `LSMEngine.start()` replays WAL records newer than existing segments.
4. MemTable is restored.
5. Later inserts can trigger flush.
6. Recovered vectors become searchable after segment registration and reader refresh.

### Coordinator Failover

1. Three coordinators participate in leader election.
2. Current leader refreshes `/vecscaledb/leader`.
3. If leader dies, the leader key expires.
4. A follower campaigns and becomes the new leader.
5. Stale local leader flags are cleared by checking etcd.
6. Writer/readers retry across all configured coordinator URLs.

## 4. File-by-File Reference

### Phase 6 Files

- `docker-compose.yml`
  - Enables `reader-0` through `reader-4`.
  - Sets full-replication reader config.
  - Sets coordinator `VECSCALE_SEARCH_FANOUT_MODE=replicated`.

- `vecscaledb/bench/load_dataset.py`
  - Loads SIFT HDF5 datasets.
  - Streams training vectors in batches without loading the full file.
  - Can bulk-insert vectors into writer.

- `vecscaledb/bench/benchmark_qps.py`
  - Runs sustained concurrent search load.
  - Reports total queries, errors, QPS, p50, p95, and p99 latency.

- `vecscaledb/reader/search.py`
  - Supports `load_all_shards=True` for replicated readers.

- `vecscaledb/coordinator/app.py`
  - Supports `scatter` and `replicated` search fanout.

### Phase 7 Files

- `vecscaledb/reader/app.py`
  - Adds `/ready`.
  - Marks ready only after initial segment refresh.
  - Keeps background refresh alive across transient coordinator failures.

- `vecscaledb/writer/app.py`
  - Keeps background registration alive across transient coordinator failures.

- `vecscaledb/coordinator/client.py`
  - Retries/fails over across configured coordinator URLs.

- `vecscaledb/coordinator/app.py`
  - Refreshes leader TTL.
  - Clears stale local leader state.

- `tests/unit/test_coordinator_client.py`
  - Verifies coordinator HTTP failover.

## 5. How To Run

Run from project root:

```powershell
cd "C:\Users\priya\OneDrive\Desktop\SEM 4\Distri\distriProject\Distributed-Vector-DB"
```

### Full Five-Reader Stack

```powershell
docker compose up -d --build
```

Check coordinators:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
```

Check readers:

```powershell
Invoke-RestMethod http://127.0.0.1:8200/ready
Invoke-RestMethod http://127.0.0.1:8201/ready
Invoke-RestMethod http://127.0.0.1:8202/ready
Invoke-RestMethod http://127.0.0.1:8203/ready
Invoke-RestMethod http://127.0.0.1:8204/ready
```

## 6. Load SIFT Data

Download dataset:

```powershell
bash data/download_sift.sh
```

Load SIFT1M into writer:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.load_dataset --dataset data\sift1m\sift1m.hdf5 --writer-url http://127.0.0.1:8100 --batch-size 10000
```

Check segment registration:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/segments
```

## 7. Run QPS Benchmark

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.benchmark_qps --dataset data\sift1m\sift1m.hdf5 --coordinator-url http://127.0.0.1:8000 --concurrency 32 --duration 60
```

## 8. Fault-Tolerance Commands

### Reader Restart

```powershell
docker compose stop reader-2
docker compose start reader-2
Invoke-RestMethod http://127.0.0.1:8202/ready
```

### Writer Crash

```powershell
docker compose kill writer-0
docker compose start writer-0
Invoke-RestMethod http://127.0.0.1:8100/health
```

### Coordinator Failover

```powershell
docker compose kill coordinator-0
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
docker compose start coordinator-0
```

## 9. How To Test

### Phase 6 Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_bench_phase6.py tests\unit\test_reader_search.py tests\integration\test_coordinator_app.py -v
```

### Phase 7 Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_coordinator_client.py tests\unit\test_writer_pipeline.py tests\integration\test_reader_app.py tests\integration\test_coordinator_app.py -v
```

### Full Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 10. Verification Status

Verified locally:

- 5 readers register and report `/ready`.
- Coordinator exact search returns expected vector ID.
- Short QPS smoke completed with zero errors.
- Reader restart recovers readiness and membership.
- Writer crash recovery preserves unflushed acknowledged vectors.
- Coordinator failover elects one new leader.
- Writer insert and coordinator search work after failover.

## 11. Known Boundaries

- Full SIFT1M QPS/recall numbers require `data/sift1m/sift1m.hdf5`.
- Phase 6 currently uses full replication, not partitioned segment ownership.
- Chaos testing is not continuous by default; manual stop/kill commands are used for verification.
