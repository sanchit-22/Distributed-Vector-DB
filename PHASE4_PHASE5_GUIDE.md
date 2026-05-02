# VecScaleDB Phase 4/5 Guide

This document explains:

- what Phase 4 and Phase 5 add,
- how writer and reader services work,
- how distributed search flows through the coordinator,
- how to run and test these phases.

Scope: current repository state under `Distributed-Vector-DB`.

## 1. High-Level Architecture

### Phase 4: Writer Node

Phase 4 separates ingestion into a dedicated Writer service.

Implemented features:

- Writer FastAPI app.
- Writer pipeline around the Phase 2 `LSMEngine`.
- WAL-backed insert/delete handling.
- Segment flush registration with coordinator.
- Background reconciliation for missed segment registrations.
- Writer health endpoint.

### Phase 5: Reader Node + Coordinator Search

Phase 5 adds stateless Readers and coordinator-facing distributed search.

Implemented features:

- Reader FastAPI app.
- Reader segment cache.
- Background segment refresh from coordinator metadata.
- Snapshot-aware reader search.
- Coordinator `/search` entry point.
- Scatter-gather search mode.
- Partial reader failure handling with `incomplete: true`.

## 2. Service Roles

### Coordinator

The coordinator owns cluster metadata:

- registered nodes,
- registered segments,
- current snapshot ID,
- reader routing,
- client-facing distributed search.

### Writer

The writer owns ingestion:

- receives inserts/deletes,
- writes WAL,
- updates MemTable,
- flushes segments,
- registers segments with coordinator.

### Reader

Readers own query execution:

- register themselves with coordinator,
- load segment metadata from coordinator,
- load segment indexes from shared storage,
- search local cached segments,
- return top-k results to coordinator.

## 3. Phase 4 Data Flow

### Writer Startup

1. Writer creates `WriterPipeline`.
2. Pipeline registers writer node with coordinator.
3. Pipeline starts `LSMEngine`.
4. Existing local segments are loaded.
5. WAL recovery restores unflushed writes.
6. Any local segment missing in coordinator metadata is registered.

### Insert Path

1. Client sends `POST /insert` to writer.
2. Writer validates ID/vector lengths and vector dimension.
3. `LSMEngine.insert(...)` appends to WAL first.
4. MemTable is updated after WAL fsync.
5. If flush occurs, new segment is registered with coordinator.
6. Response includes LSN, inserted count, flush flag, total count, segment count, and WAL LSN.

### Delete Path

1. Client sends `POST /delete` to writer.
2. Writer appends delete tombstones to WAL.
3. MemTable hides deleted IDs.
4. Response returns deleted count and current LSN.

### Segment Reconciliation

The writer has a background loop that:

- refreshes writer registration,
- registers unregistered local segments,
- removes stale coordinator segment metadata after compaction.

## 4. Phase 5 Data Flow

### Reader Startup

1. Reader creates a `CoordinatorClient`.
2. Reader creates a `SegmentCache`.
3. Reader registers itself with coordinator.
4. Reader fetches all segment metadata.
5. Reader loads relevant segment indexes from shared storage.
6. Reader starts a 5-second refresh loop.

### Reader Search

1. Coordinator calls reader `POST /search`.
2. Request includes query, `top_k`, `nprobe`, and `snapshot_id`.
3. Reader searches cached segments visible at that snapshot.
4. Results are merged/deduplicated locally.
5. Reader returns `SearchResult`.

### Coordinator Search

1. Client sends `POST /search` to coordinator.
2. Coordinator reads current snapshot ID.
3. Coordinator selects readers.
4. Coordinator sends search requests to readers.
5. Coordinator merges all reader results.
6. If any reader fails, coordinator returns partial results with `incomplete: true`.

## 5. File-by-File Reference

### Writer

- `vecscaledb/writer/app.py`
  - Writer FastAPI app.
  - Endpoints:
    - `POST /insert`
    - `POST /delete`
    - `GET /health`
  - Runs background registration/reconciliation loop.

- `vecscaledb/writer/pipeline.py`
  - Writer pipeline.
  - Owns `LSMEngine`.
  - Registers writer and segments with coordinator.
  - Handles recovery and metadata reconciliation.

### Reader

- `vecscaledb/reader/app.py`
  - Reader FastAPI app.
  - Endpoints:
    - `POST /search`
    - `GET /health`
  - Registers reader and refreshes segment cache.

- `vecscaledb/reader/search.py`
  - `SegmentCache` implementation.
  - Loads segments from shared storage.
  - Supports shard-filtered and full-replicated reader modes.
  - Performs local top-k merge.

### Coordinator Client

- `vecscaledb/coordinator/client.py`
  - HTTP client used by writer/reader services.
  - Retries across configured coordinator URLs.
  - Supports node registration, segment registration/deletion, and metadata reads.

### Shared Models

- `vecscaledb/models.py`
  - Adds `DeleteRequest`.
  - Adds `ReaderSearchRequest`, which extends search with `snapshot_id`.

### Docker

- `docker-compose.yml`
  - Defines:
    - `etcd`
    - `coordinator-0`
    - `coordinator-1`
    - `coordinator-2`
    - `writer-0`
    - `reader-0`
    - `reader-1` through `reader-4` for Phase 6+

## 6. Main Endpoints

### Writer

```text
POST http://127.0.0.1:8100/insert
POST http://127.0.0.1:8100/delete
GET  http://127.0.0.1:8100/health
```

### Reader

```text
POST http://127.0.0.1:8200/search
GET  http://127.0.0.1:8200/health
```

For additional readers:

```text
reader-1 -> http://127.0.0.1:8201
reader-2 -> http://127.0.0.1:8202
reader-3 -> http://127.0.0.1:8203
reader-4 -> http://127.0.0.1:8204
```

### Coordinator

```text
POST http://127.0.0.1:8000/search
GET  http://127.0.0.1:8000/segments
GET  http://127.0.0.1:8000/health
```

## 7. Example Requests

### Insert

```powershell
$body = @{
  ids = @(1, 2)
  vectors = @(
    @(1.0, 0.0, 0.0),
    @(0.0, 1.0, 0.0)
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8100/insert -Body $body -ContentType "application/json"
```

Use 128-dimensional vectors with the default config.

### Search Through Coordinator

```powershell
$query = @(1.0) + @(0.0) * 127
$body = @{
  query = $query
  top_k = 10
  nprobe = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -Body $body -ContentType "application/json"
```

## 8. How To Run

Run from project root:

```powershell
cd "C:\Users\priya\OneDrive\Desktop\SEM 4\Distri\distriProject\Distributed-Vector-DB"
```

### Phase 4 Stack

```powershell
docker compose up -d --build etcd coordinator-0 coordinator-1 coordinator-2 writer-0
```

Check writer:

```powershell
Invoke-RestMethod http://127.0.0.1:8100/health
```

### Phase 5 Stack

```powershell
docker compose up -d --build etcd coordinator-0 coordinator-1 coordinator-2 writer-0 reader-0
```

Check reader:

```powershell
Invoke-RestMethod http://127.0.0.1:8200/health
```

Check coordinator:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/segments
```

### Full Current Stack

The current compose file also includes five readers:

```powershell
docker compose up -d --build
```

## 9. How To Test

### Writer Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_writer_pipeline.py tests\integration\test_writer_app.py -v
```

### Reader Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_reader_search.py tests\integration\test_reader_app.py -v
```

### Coordinator Search Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_coordinator_app.py -v
```

### Full Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 10. Runtime Verification Checklist

### Writer

- `GET /health` returns `status: ok`.
- Insert below flush threshold updates `ntotal`.
- Insert above flush threshold creates at least one segment.
- Coordinator `/segments` lists flushed segment metadata.
- Restart writer; WAL recovery restores acknowledged writes.
- Delete hides IDs from search.

### Reader

- `GET /health` returns `status: ok`.
- `segments_loaded` increases after writer flush and coordinator registration.
- Direct reader search returns expected IDs.
- Reader rejects bad query dimensions.

### Coordinator Search

- Coordinator `/search` returns merged top-k results.
- Exact vector search returns distance `0.0`.
- If a reader is down, coordinator returns `incomplete: true` instead of crashing.
- After reader restart, coordinator search returns `incomplete: false`.

## 11. Implemented Edge Cases

- Writer validates vector dimensionality.
- Writer registration loop survives temporary coordinator failures.
- Writer recovers and registers existing local segments.
- Reader registration loop survives temporary coordinator failures.
- Reader cache evicts removed/stale segments.
- Reader search respects snapshot IDs.
- Coordinator search handles partial reader failure.
- Coordinator search can use scatter mode or replicated-reader mode.

## 12. Known Boundaries

- Phase 4 has one writer.
- Phase 5 search is best-effort when readers fail.
- Phase 5 introduces distributed search, but full QPS benchmarking and five-reader scaling are Phase 6.
- Segment ownership is still simple; partitioned multi-shard segment assignment is an optional later extension.
