# VecScaleDB Phase 2/3 Guide

This document explains:

- what Phase 2 and Phase 3 add,
- how the storage and coordinator layers work,
- what files/modules are involved,
- how to run and test the implementation.

Scope: current repository state under `Distributed-Vector-DB`.

## 1. High-Level Architecture

### Phase 2: Durable Single-Node Storage

Phase 2 upgrades the Phase 1 single-node API from simple in-memory buffering to a WAL-backed LSM-style storage engine.

Implemented features:

- Binary write-ahead log before acknowledging inserts/deletes.
- MemTable with per-write LSN tracking.
- Crash recovery from WAL.
- Immutable persisted segments.
- Segment flush when MemTable crosses threshold.
- Segment merge/compaction.
- Snapshot-aware search.
- Existing `/insert`, `/search`, and `/health` API compatibility preserved.

### Phase 3: Coordinator + etcd Metadata Layer

Phase 3 adds the coordinator service and metadata plane.

Implemented features:

- Coordinator FastAPI app.
- etcd-backed metadata store, with memory backend for tests.
- Coordinator registration for writers/readers/coordinators.
- Segment registration/listing/deletion.
- Consistent hash ring.
- Coordinator leader election.
- Follower proxying for metadata mutations.
- Search routing endpoints.

## 2. Phase 2 Data Flow

### Insert Path

1. `POST /insert` reaches `vecscaledb.node`.
2. The node delegates to `LSMEngine.insert(...)`.
3. `LSMEngine` appends the insert record to the WAL first.
4. After WAL fsync succeeds, vectors are inserted into the MemTable.
5. If the MemTable reaches `segment_flush_threshold`, it flushes to a persisted Segment.
6. Response includes the previous Phase 1 fields plus `lsn`.

### Search Path

1. `POST /search` creates a stable snapshot using the current WAL LSN.
2. MemTable search excludes writes newer than the snapshot.
3. Segment search includes only segments with `segment.snapshot_id <= snapshot_id`.
4. Results are merged and deduplicated by vector ID.
5. Smallest distance wins if duplicate IDs appear.

### Recovery Path

1. On startup, existing segments are loaded from shared storage.
2. WAL records are replayed in LSN order.
3. Valid records restore MemTable state.
4. Torn or checksum-invalid trailing records are ignored during recovery.

## 3. Phase 3 Metadata Flow

### Node Registration

1. Writer/Reader/Coordinator starts.
2. It posts `NodeInfo` to coordinator `/register`.
3. Leader coordinator stores it under `/vecscaledb/nodes/<node_id>`.
4. Coordinator rebuilds the reader ring from registered reader nodes.

### Segment Registration

1. Writer flushes a segment.
2. Writer posts `SegmentMeta` to coordinator `/segments`.
3. Coordinator stores it under `/vecscaledb/segments/<segment_id>`.
4. Coordinator updates `/vecscaledb/snapshots/current`.

### Leader Handling

- Only the coordinator leader mutates etcd metadata.
- Followers proxy mutating requests to the active leader.
- Coordinator heartbeat keeps node registration and leader metadata fresh.

## 4. File-by-File Reference

### Storage Layer

- `vecscaledb/storage/wal.py`
  - Binary WAL implementation.
  - Record format stores LSN, operation type, IDs, vectors, and CRC32.
  - Supports append, recovery, reading from an LSN, and truncation.

- `vecscaledb/storage/lsm.py`
  - Contains `MemTable`, `LSMEngine`, and `merge_results`.
  - Owns WAL replay, insert/delete, flush, merge, snapshot-aware search, and health metadata.

- `vecscaledb/storage/snapshot.py`
  - Snapshot manager.
  - Creates/release read snapshots.
  - Filters visible segments for a snapshot.

### Index Layer Changes

- `vecscaledb/index/segment.py`
  - Persists `vectors.npy` in addition to `index.faiss`, `ids.npy`, and `meta.json`.
  - Adds `load_arrays()` for merge/compaction.

- `vecscaledb/index/ivf_flat.py`
  - Keeps Faiss IVF_FLAT on Linux/Docker.
  - Uses a safer flat fallback on Windows to avoid Faiss access violations.

### Single-Node API

- `vecscaledb/node.py`
  - Phase 1 public API backed by Phase 2 `LSMEngine`.
  - Keeps `/insert`, `/search`, and `/health`.

### Coordinator Layer

- `vecscaledb/coordinator/app.py`
  - Coordinator FastAPI app.
  - Exposes route, register, segment, search, and health endpoints.

- `vecscaledb/coordinator/etcd_client.py`
  - Async etcd wrapper.
  - Includes memory backend for unit/integration tests.

- `vecscaledb/coordinator/ring.py`
  - Consistent hashing ring for reader routing.

### Shared Models/Config

- `vecscaledb/models.py`
  - Shared schemas: `InsertRequest`, `SearchRequest`, `SearchResult`, `SegmentMeta`, `NodeInfo`, `DeleteRequest`.

- `vecscaledb/config.py`
  - Adds metadata/storage-related settings.

## 5. Main Endpoints

### Single Node

- `POST /insert`
- `POST /search`
- `GET /health`

### Coordinator

- `GET /route/insert?count=N`
- `GET /route/search`
- `POST /register`
- `GET /segments`
- `POST /segments`
- `DELETE /segments/{segment_id}`
- `GET /health`

## 6. How To Run

Run from project root:

```powershell
cd "C:\Users\priya\OneDrive\Desktop\SEM 4\Distri\distriProject\Distributed-Vector-DB"
```

### Run Single-Node Phase 2 API

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.node
```

Server:

```text
http://127.0.0.1:8000
```

### Run Coordinator Stack

```powershell
docker compose up -d --build etcd coordinator-0 coordinator-1 coordinator-2
```

Check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
```

## 7. How To Test

### Phase 2 Storage Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_wal.py tests\unit\test_snapshot.py tests\unit\test_lsm.py -v
```

### Phase 3 Coordinator Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_ring.py tests\unit\test_etcd_client.py tests\integration\test_coordinator_app.py -v
```

### Regression Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_single_node.py -v
```

### Full Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 8. Implemented Edge Cases

- WAL checksum validation.
- WAL recovery stops/skips corrupted trailing records.
- Inserts are acknowledged only after WAL fsync.
- Restart recovers unflushed MemTable data.
- Search snapshots hide newer writes.
- Segment merge deduplicates IDs.
- Coordinator followers proxy metadata writes to leader.
- Coordinator search falls back to max segment snapshot if snapshot metadata is missing.
- Coordinator leader heartbeat keeps election metadata fresh.

## 9. Known Boundaries

- Deletes exist internally and through writer later, but Phase 2 single-node delete is not exposed as a public REST endpoint.
- Phase 3 does not yet ingest data through writer service; that starts in Phase 4.
- Phase 3 routing prepares distributed roles, but reader search fanout is implemented in Phase 5.
