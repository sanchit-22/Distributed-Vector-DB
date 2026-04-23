# VecScaleDB — Agent Implementation Guide

> **Project**: VecScaleDB — Distributed Vector Database  
> **Reference**: Milvus, SIGMOD 2021  
> **Stack**: Python 3.11+, FastAPI, Faiss, etcd, Docker / Docker Compose  
> **Execution Model**: Work through each phase completely. Run the verification checklist at the end of every phase before proceeding. Do not start Phase N+1 until Phase N is green.

---

## Repository Layout (target, build up to this)

```
vecscaledb/
├── agent.md                    ← this file
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── data/                       ← datasets live here (gitignored)
│   ├── sift10k/
│   └── sift1m/
├── shared_storage/             ← emulates object-store / NFS (gitignored)
├── vecscaledb/
│   ├── __init__.py
│   ├── config.py               ← central config / env vars
│   ├── models.py               ← Pydantic schemas shared across nodes
│   ├── index/
│   │   ├── __init__.py
│   │   ├── ivf_flat.py         ← IVF_FLAT wrapper around Faiss
│   │   └── segment.py          ← immutable Segment abstraction
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── wal.py              ← Write-Ahead Log
│   │   ├── lsm.py              ← LSM flush / merge logic
│   │   └── snapshot.py         ← snapshot / MVCC bookkeeping
│   ├── coordinator/
│   │   ├── __init__.py
│   │   ├── app.py              ← FastAPI app for coordinator
│   │   ├── ring.py             ← consistent hashing ring
│   │   └── etcd_client.py      ← etcd wrapper
│   ├── writer/
│   │   ├── __init__.py
│   │   ├── app.py              ← FastAPI app for writer
│   │   └── pipeline.py         ← ingest → WAL → memtable → flush
│   ├── reader/
│   │   ├── __init__.py
│   │   ├── app.py              ← FastAPI app for reader
│   │   └── search.py           ← scatter-gather query execution
│   └── bench/
│       ├── load_dataset.py
│       ├── benchmark_qps.py
│       └── recall_eval.py
└── tests/
    ├── unit/
    │   ├── test_ivf_flat.py
    │   ├── test_wal.py
    │   ├── test_snapshot.py
    │   └── test_ring.py
    └── integration/
        ├── test_single_node.py
        └── test_cluster.py
```

---

## Global Conventions

- **Language**: Python 3.11. All async I/O via `asyncio` + `httpx` for inter-node calls.
- **API style**: REST (FastAPI). All request/response bodies validated by Pydantic v2.
- **Shared storage**: a local directory (`./shared_storage/`) mounted by all containers simulates object storage. Each Segment is a directory: `shared_storage/segments/<segment_id>/`.
- **Vector IDs**: `uint64`. Use `numpy` arrays throughout; never plain Python lists for numeric work.
- **Distances**: L2 (Euclidean squared) for SIFT datasets, cosine optionally configurable.
- **Environment variables** drive all tunables; document defaults in `config.py`.
- **Logging**: Python `structlog` with JSON output. Every log line must include `node_role`, `node_id`, `trace_id`.
- **Tests**: `pytest` + `pytest-asyncio`. Minimum 80 % line coverage before phase sign-off.

---

## Phase 0 — Environment & Skeleton

### Goal
A runnable Python package, Docker Compose cluster definition, and dataset download scripts. No logic yet.

### Steps

#### 0.1 — Python project bootstrap
1. Create `pyproject.toml` using `[build-system] = hatchling`. Set `requires-python = ">=3.11"`.
2. Create `requirements.txt` with pinned versions:
   ```
   fastapi==0.111.*
   uvicorn[standard]==0.29.*
   httpx==0.27.*
   faiss-cpu==1.8.*          # use faiss-gpu if a CUDA node is available
   numpy==1.26.*
   pydantic==2.7.*
   structlog==24.*
   etcd3==0.12.*
   pytest==8.*
   pytest-asyncio==0.23.*
   h5py==3.11.*              # for reading .hdf5 benchmark files
   tqdm==4.*
   ```
3. Create `vecscaledb/__init__.py` with `__version__ = "0.1.0"`.
4. Create `vecscaledb/config.py`:
   ```python
   from pydantic_settings import BaseSettings

   class Settings(BaseSettings):
       node_role: str = "reader"           # coordinator | writer | reader
       node_id: str = "node-0"
       coordinator_urls: list[str] = ["http://coordinator-0:8000"]
       writer_url: str = "http://writer-0:8100"
       shared_storage_path: str = "/shared_storage"
       wal_path: str = "/wal"
       etcd_endpoints: list[str] = ["http://etcd:2379"]
       dim: int = 128
       nlist: int = 256                    # IVF_FLAT number of centroids
       nprobe: int = 32                    # IVF_FLAT search-time probe count
       segment_flush_threshold: int = 50_000   # vectors before flush
       segment_merge_threshold: int = 4        # segments before merge
       snapshot_ttl_seconds: int = 60

       class Config:
           env_prefix = "VECSCALE_"
   ```

#### 0.2 — Dataset download scripts
1. In `data/` create `download_sift.sh`:
   ```bash
   #!/usr/bin/env bash
   # Downloads ANN benchmark datasets as HDF5 files from ann-benchmarks.com
   mkdir -p sift10k sift1m deep1m
   wget -c http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O sift1m/sift1m.hdf5
   # SIFT10K is a slice we extract ourselves from SIFT1M
   ```
2. Create `vecscaledb/bench/load_dataset.py`:
   ```python
   import h5py, numpy as np

   def load_sift1m(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
       """Returns (train_vectors, query_vectors, ground_truth_neighbors)."""
       with h5py.File(path, "r") as f:
           train = np.array(f["train"], dtype=np.float32)
           test  = np.array(f["test"],  dtype=np.float32)
           neighbors = np.array(f["neighbors"], dtype=np.int32)
       return train, test, neighbors

   def load_sift10k(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
       train, test, neighbors = load_sift1m(path)
       return train[:10_000], test[:100], neighbors[:100]
   ```

#### 0.3 — Docker Compose skeleton
Create `docker-compose.yml` with the following services (no logic, just structural skeleton):
```yaml
version: "3.9"
services:
  etcd:
    image: bitnami/etcd:3.5
    environment:
      ALLOW_NONE_AUTHENTICATION: "yes"
    ports: ["2379:2379"]

  coordinator-0:
    build: .
    command: python -m vecscaledb.coordinator.app
    environment:
      VECSCALE_NODE_ROLE: coordinator
      VECSCALE_NODE_ID: coordinator-0
    depends_on: [etcd]
    ports: ["8000:8000"]
    volumes:
      - ./shared_storage:/shared_storage

  writer-0:
    build: .
    command: python -m vecscaledb.writer.app
    environment:
      VECSCALE_NODE_ROLE: writer
      VECSCALE_NODE_ID: writer-0
    depends_on: [coordinator-0]
    ports: ["8100:8100"]
    volumes:
      - ./shared_storage:/shared_storage
      - ./wal:/wal

  reader-0:
    build: .
    command: python -m vecscaledb.reader.app
    environment:
      VECSCALE_NODE_ROLE: reader
      VECSCALE_NODE_ID: reader-0
      VECSCALE_READER_SHARD_ID: "0"
    depends_on: [coordinator-0]
    ports: ["8200:8200"]
    volumes:
      - ./shared_storage:/shared_storage
```
Add `reader-1` through `reader-4` as replicas in later phases; keep them commented out for now.

Create a minimal `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["python", "-m", "vecscaledb.coordinator.app"]
```

#### 0.4 — Models (shared Pydantic schemas)
Create `vecscaledb/models.py`:
```python
from pydantic import BaseModel
import numpy as np

class InsertRequest(BaseModel):
    vectors: list[list[float]]   # shape [N, dim]
    ids: list[int]

class SearchRequest(BaseModel):
    query: list[float]           # shape [dim]
    top_k: int = 10
    nprobe: int = 32

class SearchResult(BaseModel):
    ids: list[int]
    distances: list[float]

class SegmentMeta(BaseModel):
    segment_id: str
    shard_id: int
    num_vectors: int
    snapshot_id: int
    path: str                    # absolute path inside shared_storage

class NodeInfo(BaseModel):
    node_id: str
    role: str                    # coordinator | writer | reader
    address: str
    shard_ids: list[int] = []
```

### Phase 0 Verification Checklist
- [ ] `pip install -e .` succeeds with no errors.
- [ ] `python -c "import vecscaledb; print(vecscaledb.__version__)"` prints `0.1.0`.
- [ ] `docker compose config` validates without errors.
- [ ] `python vecscaledb/bench/load_dataset.py` loads SIFT10K slice without exceptions.
- [ ] `pytest tests/` collects 0 tests and exits with code 0 (no failures on empty suite).

---

## Phase 1 — Core Index: IVF_FLAT + Single-Node Search API

### Goal
A single-process FastAPI server that ingests float32 vectors, builds an IVF_FLAT Faiss index, and answers k-NN queries. Verified for recall ≥ 0.95 on SIFT10K.

### Steps

#### 1.1 — IVF_FLAT wrapper (`vecscaledb/index/ivf_flat.py`)

Implement the class `IVFFlatIndex`:

```
IVFFlatIndex(dim: int, nlist: int)
    .train(vectors: np.ndarray) -> None
        # vectors shape [N, dim], N >= nlist
        # Calls faiss.IndexFlatL2 as quantizer
        # Builds faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        # Calls index.train(vectors)

    .add(vectors: np.ndarray, ids: np.ndarray) -> None
        # vectors [N, dim], ids [N] uint64
        # Wraps in faiss.IndexIDMap2 so external IDs are preserved
        # Calls index.add_with_ids(vectors, ids)

    .search(query: np.ndarray, top_k: int, nprobe: int) -> (np.ndarray, np.ndarray)
        # query [dim], returns (distances [top_k], ids [top_k])
        # Sets index.nprobe = nprobe before calling index.search

    .save(path: str) -> None
        # faiss.write_index(self.index, path)

    .load(path: str) -> None
        # self.index = faiss.read_index(path)

    @property
    .ntotal -> int
```

Important implementation details:
- Store `self._trained: bool = False`. Raise `RuntimeError` if `add()` is called before `train()`.
- Use `faiss.IndexIDMap2` wrapping `faiss.IndexIVFFlat` so you can pass external integer IDs.
- Cast all vectors to `np.float32` and call `np.ascontiguousarray()` before any Faiss call — Faiss segfaults on non-contiguous or wrong-dtype arrays.
- Return `-1` IDs from Faiss (unfilled slots) as `None` after filtering.

#### 1.2 — Segment abstraction (`vecscaledb/index/segment.py`)

A Segment is an immutable, persisted unit of storage. Implement `Segment`:

```
Segment
    segment_id: str              # uuid4 hex
    shard_id: int
    snapshot_id: int             # monotonically increasing version
    path: str                    # e.g. shared_storage/segments/<id>/
    meta: SegmentMeta

    @classmethod
    .create(shard_id, snapshot_id, vectors, ids, dim, nlist) -> Segment
        # 1. Generate segment_id = uuid4().hex
        # 2. Build directory path
        # 3. Train + add all vectors to IVFFlatIndex
        # 4. Save index to <path>/index.faiss
        # 5. Save ids as numpy array to <path>/ids.npy
        # 6. Write <path>/meta.json  (SegmentMeta serialized)
        # 7. Return Segment object

    @classmethod
    .load(path: str) -> Segment
        # Read meta.json, load index.faiss, ids.npy

    .search(query, top_k, nprobe) -> (distances, ids)
        # Delegates to self._index.search()
```

#### 1.3 — Single-node FastAPI app

For now create a single `vecscaledb/node.py` (temporary, replaced in later phases) that exposes:

```
POST /insert
    Body: InsertRequest
    - Buffer vectors in a MemTable (dict: id -> vector as np.ndarray)
    - When len(memtable) >= FLUSH_THRESHOLD, call Segment.create() and clear memtable
    - Return {"inserted": N, "segment_flushed": bool}

POST /search
    Body: SearchRequest
    - Search all loaded segments + scan memtable with brute-force (faiss.IndexFlatL2)
    - Merge results: heapq.nsmallest(top_k, combined by distance)
    - Return SearchResult

GET /health
    Return {"status": "ok", "ntotal": total_vectors}
```

#### 1.4 — Recall evaluation (`vecscaledb/bench/recall_eval.py`)

```python
def compute_recall_at_k(predicted_ids: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    predicted_ids: [N, k] - top-k results for N queries
    ground_truth:  [N, R] - true nearest neighbors (R >= k)
    Returns mean recall@k across all queries.
    """
```

Script flow:
1. Load SIFT10K.
2. POST all 10 000 train vectors via `/insert` in batches of 1 000.
3. For each of the 100 query vectors POST `/search` with `top_k=100`.
4. Compute `recall@10` and `recall@100`. Assert recall@10 ≥ 0.95.

### Phase 1 Verification Checklist
- [ ] Unit test: `test_ivf_flat.py` — train on 5 000 random float32 vectors, insert 5 000, search returns exactly `top_k` results, all IDs are within inserted range.
- [ ] Unit test: `test_segment.py` — `Segment.create()` writes files to disk; `Segment.load()` reloads and returns same search results (within floating-point tolerance).
- [ ] Integration: Load SIFT10K, insert via REST, query 100 vectors → recall@10 ≥ 0.95.
- [ ] `GET /health` returns `ntotal = 10000`.
- [ ] `pytest tests/unit/test_ivf_flat.py tests/unit/test_segment.py` — all green.

---

## Phase 2 — Write-Ahead Log + LSM-Inspired Flush/Merge + Snapshot Isolation

### Goal
Durability: no write is acknowledged until it is recorded in the WAL. Snapshot isolation: a search sees a stable, point-in-time view even while concurrent inserts flush new segments. Segment merging keeps read-path cost bounded.

### Steps

#### 2.1 — Write-Ahead Log (`vecscaledb/storage/wal.py`)

The WAL is an append-only binary log on local disk.

**Log record format** (write sequentially, read back on recovery):

| Field | Type | Size |
|---|---|---|
| LSN (Log Sequence Number) | uint64 | 8 bytes |
| op_type | uint8 (0=INSERT, 1=DELETE) | 1 byte |
| num_vectors | uint32 | 4 bytes |
| id_array | num_vectors × uint64 | variable |
| vector_array | num_vectors × dim × float32 | variable |
| checksum | uint32 (CRC32 of above) | 4 bytes |

Implement `WAL`:
```
WAL(path: str, dim: int)
    .open() -> None            # open file in "ab" + "rb" mode; load last LSN from index file
    .append(op_type, ids, vectors) -> int   # returns LSN; fsync after write
    .read_from(lsn: int) -> Iterator[WALRecord]
    .truncate_before(lsn: int) -> None      # delete old records after checkpoint
    .close() -> None
    .recover() -> list[WALRecord]           # replay all records, return them ordered by LSN
```

Use `struct.pack` / `struct.unpack` for serialization — no pickle, no JSON (too slow for hot path).

#### 2.2 — MemTable

Implement `MemTable` in `vecscaledb/storage/lsm.py`:
```
MemTable(dim: int, flush_threshold: int)
    .insert(ids: np.ndarray, vectors: np.ndarray) -> None
    .delete(ids: np.ndarray) -> None       # mark as tombstone; exclude from search
    .search(query: np.ndarray, top_k: int) -> (distances, ids)
        # Brute-force: faiss.IndexFlatL2 rebuilt each time (acceptable for small memtable)
    .should_flush() -> bool                # len(self._live) >= flush_threshold
    .to_arrays() -> (np.ndarray, np.ndarray)   # returns (ids, vectors) excluding tombstones
    .clear() -> None
```

#### 2.3 — LSMEngine (`vecscaledb/storage/lsm.py`)

```
LSMEngine(shared_storage: str, wal_path: str, shard_id: int,
          dim: int, nlist: int, flush_threshold: int, merge_threshold: int)

    .start() -> None
        # Replay WAL → rebuild MemTable state
        # Load all existing Segments from shared_storage for this shard

    .insert(ids, vectors) -> int    # returns LSN
        # 1. WAL.append(INSERT, ids, vectors)
        # 2. MemTable.insert(ids, vectors)
        # 3. If MemTable.should_flush(): asyncio.create_task(_flush())
        # 4. Return LSN

    .delete(ids) -> int
        # 1. WAL.append(DELETE, ids, zeros)
        # 2. MemTable.delete(ids)

    async def _flush() -> Segment
        # 1. Snapshot current MemTable → call to_arrays()
        # 2. Clear MemTable AFTER snapshot (no lost writes)
        # 3. Create new snapshot_id = self._next_snapshot_id++
        # 4. Segment.create(shard_id, snapshot_id, vectors, ids, ...)
        # 5. Append to self._segments list
        # 6. WAL.truncate_before(current_lsn - WAL_RETAIN_COUNT)
        # 7. If len(self._segments) >= merge_threshold: asyncio.create_task(_merge())
        # 8. Return new Segment

    async def _merge() -> Segment
        # 1. Take the oldest merge_threshold segments
        # 2. Load all their (ids, vectors) into memory
        # 3. Create one merged Segment with a new snapshot_id
        # 4. Replace old segments in self._segments
        # 5. Delete old segment directories from disk

    .search(query, top_k, nprobe, snapshot_id=None) -> (distances, ids)
        # See 2.4 below
```

#### 2.4 — Snapshot Isolation (`vecscaledb/storage/snapshot.py`)

Implement a `SnapshotManager`:
```
SnapshotManager
    .create_snapshot() -> int
        # Atomically read current snapshot_id counter → return it
        # Register it as an active reader snapshot

    .release_snapshot(snapshot_id: int) -> None
        # De-register; if no readers hold this snapshot, allow GC

    .visible_segments(snapshot_id: int, all_segments: list[Segment]) -> list[Segment]
        # Return only segments whose snapshot_id <= given snapshot_id
        # This ensures a query never sees data that was inserted after it started
```

Integration with search:
- On `POST /search`, reader calls `SnapshotManager.create_snapshot()` → gets `sid`.
- Passes `sid` to `LSMEngine.search()`.
- `LSMEngine.search()` calls `SnapshotManager.visible_segments(sid, self._segments)` to filter.
- After result is assembled, calls `SnapshotManager.release_snapshot(sid)`.

**Search merge logic** (critical — get this right):
```
def _merge_results(segment_results: list[tuple], memtable_result: tuple, top_k: int):
    # Each result is (distances[k], ids[k])
    # Flatten all into one list of (distance, id) pairs
    # Deduplicate by id (keep minimum distance)
    # Return top_k by ascending distance
```

#### 2.5 — WAL Recovery test

Write `tests/unit/test_wal.py`:
1. Create WAL, append 3 records.
2. Close WAL (simulate crash).
3. Reopen WAL, call `recover()`.
4. Assert all 3 records returned in LSN order with correct checksums.
5. Corrupt the last 4 bytes of the last record (simulate torn write).
6. Re-run recovery — assert only 2 records returned (corrupted record skipped).

### Phase 2 Verification Checklist
- [ ] `test_wal.py` — all cases including crash/corruption pass.
- [ ] `test_snapshot.py` — concurrent insert + search scenario: reader sees only pre-insert data when snapshot was taken before insert completed.
- [ ] WAL recovery: kill the node process mid-insert batch; restart; verify all fully-written records are recovered, partial record is discarded.
- [ ] Segment merge: insert 4 × `flush_threshold` vectors; verify exactly 1 merged segment exists on disk after stabilization (not 4 separate ones).
- [ ] Recall@10 on SIFT10K still ≥ 0.95 after adding the WAL/LSM layer.

---

## Phase 3 — Coordinator: Consistent Hashing + etcd HA

### Goal
A highly available coordinator cluster (3 replicas behind a single VIP or DNS name) that maps vector IDs to shards via a consistent hashing ring and maintains cluster membership in etcd.

### Steps

#### 3.1 — Consistent Hashing Ring (`vecscaledb/coordinator/ring.py`)

Implement `ConsistentHashRing`:
```
ConsistentHashRing(virtual_nodes: int = 150)
    .add_node(node_id: str) -> None
        # Add virtual_nodes virtual replicas to the ring
        # Key: hash(f"{node_id}:{i}") for i in range(virtual_nodes)
        # Use xxhash (or hashlib.md5) for speed, not Python hash()

    .remove_node(node_id: str) -> None

    .get_node(key: int) -> str
        # Hash the key, walk ring clockwise to find owner node_id

    .get_nodes_for_range(start_id: int, end_id: int) -> set[str]
        # Return all node_ids that own any key in [start_id, end_id]

    def get_shard_id(node_id: str) -> int
        # Deterministic shard integer: hash(node_id) % total_shards
```

Test with: 5 nodes, insert 100 000 random keys → distribution should be within ±20% of equal (validate load balance).

#### 3.2 — etcd client wrapper (`vecscaledb/coordinator/etcd_client.py`)

```
EtcdClient(endpoints: list[str])
    async def put(key: str, value: str, ttl: int = None) -> None
    async def get(key: str) -> str | None
    async def get_prefix(prefix: str) -> dict[str, str]
    async def delete(key: str) -> None
    async def watch_prefix(prefix: str) -> AsyncIterator[tuple[str, str]]
    async def campaign(election_name: str, value: str) -> bool
        # Leader election: returns True if this node became leader
    async def observe_leader(election_name: str) -> AsyncIterator[str]
```

Use the `etcd3` library. Wrap all calls in retry logic with exponential backoff (max 3 retries, base delay 0.1 s).

etcd key namespaces:
```
/vecscaledb/nodes/<node_id>       → NodeInfo JSON  (TTL 10s, refreshed by heartbeat)
/vecscaledb/segments/<segment_id> → SegmentMeta JSON
/vecscaledb/snapshots/current     → current snapshot_id (uint64 as string)
/vecscaledb/leader                → active coordinator leader node_id
```

#### 3.3 — Coordinator FastAPI app (`vecscaledb/coordinator/app.py`)

```
Startup:
    1. Connect to etcd.
    2. Run leader election (campaign). The leader handles all writes to etcd ring state.
    3. Build ConsistentHashRing from existing /vecscaledb/nodes/ entries.
    4. Subscribe to /vecscaledb/nodes/ prefix watch — update ring on changes.
    5. Start heartbeat task: PUT /vecscaledb/nodes/<node_id> every 5 s (TTL 10 s).

Endpoints:
    GET /route/insert?count=N
        # Returns writer_url (always the single Writer node)
        # Returns the shard_id for this batch (ring.get_node(random_key_in_batch))

    GET /route/search?query_hash=H
        # Returns list[NodeInfo] — reader nodes that own shards relevant to H
        # For a full-index scan (no partition key given), return ALL reader nodes

    POST /register
        Body: NodeInfo
        # Writer and Readers call this on startup
        # Coordinator stores in etcd + updates ring

    GET /segments
        # Returns all SegmentMeta from etcd

    POST /segments
        Body: SegmentMeta
        # Writer calls this after each flush

    GET /health
        # Returns {"leader": bool, "ring_size": N, "nodes": [...]}
```

#### 3.4 — Coordinator HA (3 replicas)
Add `coordinator-1` and `coordinator-2` to `docker-compose.yml` (same image, different `NODE_ID`). All three participate in the etcd leader election. Only the leader accepts write mutations to etcd; followers proxy to leader. Clients always talk to `coordinator-0` (or use a round-robin list).

### Phase 3 Verification Checklist
- [ ] `test_ring.py` — 5 nodes, 100 000 keys → max deviation from equal share < 25%.
- [ ] Ring rebalance: remove a node → keys redistribute; add it back → same keys return to it.
- [ ] etcd integration: `docker compose up etcd coordinator-0 coordinator-1 coordinator-2`; kill `coordinator-0`; within 10 s a follower becomes leader (verify via `GET /health`).
- [ ] `POST /register` from a new reader → appears in `GET /health` nodes list.
- [ ] `GET /route/search` returns all currently registered reader nodes.

---

## Phase 4 — Writer Node

### Goal
The single Writer node: receives inserts via REST, writes WAL, manages the MemTable, flushes immutable Segments to shared storage, and registers Segments with the Coordinator.

### Steps

#### 4.1 — Writer pipeline (`vecscaledb/writer/pipeline.py`)

```
WriterPipeline(config: Settings, coordinator_client: httpx.AsyncClient)
    async def start() -> None
        # 1. Register self with coordinator: POST coordinator/register
        # 2. Initialize LSMEngine (which replays WAL)
        # 3. Recover any un-registered segments from shared_storage
        #    (flush might have succeeded but coordinator registration crashed)

    async def insert(ids, vectors) -> dict
        # 1. Validate: len(ids) == len(vectors), dim matches config
        # 2. lsm_engine.insert(ids, vectors)  → returns LSN
        # 3. If a flush completed since last call, register new segment:
        #       POST coordinator/segments with SegmentMeta
        # 4. Return {"lsn": lsn, "ntotal": lsm_engine.total_vectors}

    async def delete(ids) -> dict
        # lsm_engine.delete(ids)
        # Return {"deleted": len(ids)}
```

#### 4.2 — Writer FastAPI app (`vecscaledb/writer/app.py`)

```
POST /insert
    Body: InsertRequest
    → pipeline.insert(ids, vectors)

POST /delete
    Body: {"ids": [int, ...]}
    → pipeline.delete(ids)

GET /health
    → {"status": "ok", "ntotal": N, "segments": K, "wal_lsn": L}
```

Add a background task that polls `pipeline.lsm_engine._segments` and registers any newly flushed segments with the coordinator (in case the inline registration in `insert()` misses one due to timing).

#### 4.3 — Coordinator client helper

Create `vecscaledb/coordinator/client.py`:
```
CoordinatorClient(coordinator_urls: list[str])
    async def register_node(info: NodeInfo) -> None
    async def register_segment(meta: SegmentMeta) -> None
    async def get_all_segments() -> list[SegmentMeta]
    async def get_search_targets() -> list[NodeInfo]  # reader nodes
    # Implement retry across coordinator_urls list (failover to replicas)
```

### Phase 4 Verification Checklist
- [ ] `docker compose up etcd coordinator-0 writer-0` starts cleanly.
- [ ] `POST /insert` 1 000 vectors → `GET /health` shows `ntotal=1000`.
- [ ] Insert 60 000 vectors (> flush_threshold=50 000) → `GET /health` shows `segments=1`, `ntotal=60000`.
- [ ] Coordinator `GET /segments` lists the flushed segment.
- [ ] Kill writer mid-insert, restart → WAL recovery restores all acknowledged writes; `ntotal` is consistent.
- [ ] `POST /delete` removes IDs from MemTable (verify via search returning no results for deleted IDs).

---

## Phase 5 — Reader Node + Scatter-Gather Search

### Goal
Stateless Reader nodes pull Segments from shared storage, execute local k-NN queries, and participate in scatter-gather fan-out from the Coordinator. A client-facing search entry-point at the Coordinator fans out to all Readers and merges results.

### Steps

#### 5.1 — Reader segment loader (`vecscaledb/reader/search.py`)

```
SegmentCache(shared_storage: str, shard_id: int, max_segments_in_memory: int = 20)
    # Uses an LRU cache keyed by segment_id

    async def refresh(all_segments: list[SegmentMeta]) -> None
        # Compare with coordinator's segment list
        # Load new segments not yet in cache
        # Evict segments no longer in coordinator's list

    def search_all(query, top_k, nprobe, snapshot_id) -> (distances, ids)
        # Search all cached segments for this shard
        # Merge results with merge_results() helper from Phase 2
```

#### 5.2 — Reader FastAPI app (`vecscaledb/reader/app.py`)

```
Startup:
    1. Register with coordinator.
    2. Start background task: every 5 s call coordinator GET /segments → refresh cache.

POST /search
    Body: SearchRequest + {"snapshot_id": int}
    → cache.search_all(query, top_k, nprobe, snapshot_id)
    → Return SearchResult

GET /health
    → {"shard_id": N, "segments_loaded": K, "ntotal": M}
```

#### 5.3 — Scatter-Gather at Coordinator (`vecscaledb/coordinator/app.py` addition)

Add to coordinator:
```
POST /search
    Body: SearchRequest
    1. snapshot_id = etcd.get("/vecscaledb/snapshots/current")
    2. readers = etcd.get_prefix("/vecscaledb/nodes/reader-*")
    3. Fan out: for each reader, POST reader_url/search with body + snapshot_id
       (Use asyncio.gather for parallel fan-out)
    4. Collect all SearchResult responses
    5. Merge: flatten all (distance, id) pairs, deduplicate, return top_k
    6. Return merged SearchResult
```

Handle partial failures: if a reader is unreachable, log a warning and continue with remaining readers (best-effort). Include an `"incomplete": true` flag in the response if any reader was skipped.

#### 5.4 — Snapshot propagation

When Writer flushes a new segment and registers it with the Coordinator:
1. Coordinator increments `/vecscaledb/snapshots/current` atomically in etcd.
2. All Readers' background refresh tasks pick up the new segment within their 5 s polling window.
3. New searches issued after the coordinator increments the snapshot will include the new segment.

### Phase 5 Verification Checklist
- [ ] `docker compose up` (all services: etcd + 3 coordinators + 1 writer + 1 reader).
- [ ] Insert 1 000 vectors via Writer. Reader `GET /health` shows `ntotal=1000` within 10 s.
- [ ] `POST coordinator/search` returns correct top-10 results (cross-reference with brute-force).
- [ ] Recall@10 on SIFT10K via the full distributed path ≥ 0.95.
- [ ] Kill reader mid-search → coordinator returns `"incomplete": true` but does not crash.
- [ ] Snapshot isolation: insert batch A, take snapshot S1; insert batch B; search with S1 → batch B results invisible.

---

## Phase 6 — Horizontal Scaling: 1 → 5 Readers (QPS Benchmark)

### Goal
Reproduce Milvus Figure 10b: QPS scales near-linearly as readers increase from 1 to 5. Generate a QPS-vs-nodes curve.

### Steps

#### 6.1 — Load dataset at scale

Insert all 1 000 000 SIFT1M vectors into the Writer in batches of 10 000. Script: `vecscaledb/bench/load_dataset.py`.

Expect ~20 segments (1M / 50K flush threshold). Verify via `coordinator GET /segments`.

#### 6.2 — Multi-reader Docker Compose

Uncomment and configure `reader-1` through `reader-4` in `docker-compose.yml`.

Each reader is assigned a `VECSCALE_READER_SHARD_ID`. For a 5-reader cluster, all readers load all segments (no partitioning yet — full replication, rely on Coordinator to fan-out to one reader per search). Partitioned sharding is an optional extension; full-replication is the baseline for the QPS experiment.

#### 6.3 — QPS benchmark script (`vecscaledb/bench/benchmark_qps.py`)

```python
"""
Runs a sustained query load at concurrency=C for T seconds.
Reports: total_queries, elapsed, QPS, p50/p95/p99 latency.
"""
import asyncio, time, numpy as np, httpx, statistics

async def run_benchmark(
    coordinator_url: str,
    queries: np.ndarray,         # [N, dim] float32
    top_k: int,
    nprobe: int,
    concurrency: int,
    duration_seconds: float,
) -> dict:
    # Use a semaphore of size `concurrency`
    # Each coroutine: pick random query, POST /search, record latency
    # Run until duration_seconds elapsed
    # Compute QPS = total_queries / elapsed
    # Compute percentiles from latency list
    ...
```

Run with:
- 1 reader, concurrency=32, 60 s
- 2 readers, concurrency=32, 60 s
- 3 readers, concurrency=32, 60 s
- 5 readers, concurrency=32, 60 s

Plot QPS-vs-nodes. Expected shape: near-linear up to 3–4 nodes, then slightly sub-linear at 5 (coordinator scatter-gather overhead).

#### 6.4 — Shard partitioning (optional but recommended)

To avoid duplicating search work across all 5 readers:
1. During segment creation, Writer assigns `shard_id = segment_count % num_readers`.
2. Coordinator scatter-gather sends each query only to the Reader that owns the segment's shard.
3. Each Reader loads only its own shard's segments → 5× less memory per node.

Implement by updating `ConsistentHashRing.get_node(segment_id)` and the Coordinator's scatter-gather logic.

### Phase 6 Verification Checklist
- [ ] SIFT1M fully loaded: `coordinator GET /segments` shows ≥ 18 segments.
- [ ] 1-reader QPS baseline established (record the number).
- [ ] 5-reader QPS ≥ 3.5 × 1-reader QPS (≥ 70% linear scaling efficiency).
- [ ] Recall@10 on SIFT1M via 5-reader cluster ≥ 0.90 (slightly lower than SIFT10K due to quantization at scale).
- [ ] Latency p99 < 100 ms for 32-concurrent queries on the 5-reader cluster.

---

## Phase 7 — Fault Tolerance

### Goal
The system handles node crashes gracefully: reader restarts load segments from shared storage without data loss; writer restarts recover from WAL; coordinator failover is handled by etcd leader election.

### Steps

#### 7.1 — Reader restart recovery

1. Kill `reader-2` container (`docker compose stop reader-2`).
2. Send 100 queries via coordinator while `reader-2` is down.
3. Verify: coordinator returns `"incomplete": true` for queries that needed `reader-2`'s shards, but does not return 500.
4. Restart `reader-2` (`docker compose start reader-2`).
5. Within 30 s: Reader re-registers, refreshes segment cache from coordinator.
6. Send 100 more queries → all complete, `"incomplete"` disappears.

Implement: Reader startup sequence must always call `CoordinatorClient.register_node()` and `cache.refresh()` before serving traffic. Add a `/ready` endpoint that returns 200 only after the first cache refresh completes. Add `healthcheck` in Docker Compose using `/ready`.

#### 7.2 — Writer crash recovery

1. Insert 30 000 vectors (WAL written, not yet flushed — below 50 K threshold).
2. `docker compose kill writer-0`.
3. `docker compose start writer-0`.
4. On startup, `WriterPipeline.start()` replays WAL → MemTable rebuilt with 30 000 vectors.
5. Insert 20 001 more vectors → flush triggered → segment registered with coordinator.
6. Reader picks up new segment. Search returns all 50 001 vectors.

Verify: No acknowledged write was lost (all writes the Writer returned a LSN for must appear in search results after recovery).

#### 7.3 — Coordinator failover

1. Three coordinator replicas running.
2. `docker compose kill coordinator-0` (the current leader).
3. Within 15 s: `coordinator-1` or `coordinator-2` wins election (verify via `/health` `leader: true`).
4. Send search and insert requests to any coordinator URL → they work.

Implement: Writer and Readers use the `CoordinatorClient` with a list of all coordinator URLs and retry across them.

#### 7.4 — Chaos test (optional)

Write a script `tests/integration/test_chaos.py`:
- Randomly kill one of the reader nodes every 30 s.
- Continuously run searches.
- Assert: QPS drops at most 30% when one of 5 readers is down.
- Assert: Zero searches return a 5xx error (only `incomplete: true` is acceptable).

### Phase 7 Verification Checklist
- [ ] Reader restart: zero data loss, re-registered within 30 s, `incomplete` flag gone.
- [ ] Writer WAL recovery: all acknowledged writes survive crash. Verified by diff of inserted-IDs vs search results.
- [ ] Coordinator failover: new leader elected within 15 s, requests succeed on failover URL.
- [ ] `docker compose up` (full cluster) followed by `pytest tests/integration/` — all green.

---

## Phase 8 — Observability & Final Evaluation

### Goal
Structured logs, per-endpoint metrics, and a final benchmark report reproducing Milvus Figure 10b.

### Steps

#### 8.1 — Structured logging

In every FastAPI app, configure `structlog`:
```python
import structlog
logger = structlog.get_logger().bind(node_role=config.node_role, node_id=config.node_id)
```

Every request handler must log at entry and exit:
```python
logger.info("search.start", top_k=req.top_k, nprobe=req.nprobe, trace_id=trace_id)
logger.info("search.end", duration_ms=elapsed, results=len(ids), trace_id=trace_id)
```

#### 8.2 — Prometheus metrics (optional but recommended)

Add `prometheus-fastapi-instrumentator` to requirements. Expose `/metrics` on each node. Track:
- `vecscaledb_insert_total` (counter, labels: node_id)
- `vecscaledb_search_duration_seconds` (histogram, labels: node_role, node_id)
- `vecscaledb_segments_loaded` (gauge, labels: node_id)
- `vecscaledb_wal_lsn` (gauge, labels: node_id)

#### 8.3 — Final benchmark report

Run the complete benchmark suite and generate `benchmark_report.md`:

```markdown
# VecScaleDB Benchmark Report

## Dataset: SIFT1M (1M vectors, 128 dim, L2)

## Recall Evaluation
| Config           | Recall@10 | Recall@100 |
|------------------|-----------|------------|
| Single node      |           |            |
| 5-reader cluster |           |            |

## QPS Scaling (nprobe=32, concurrency=32, duration=60s)
| Readers | QPS   | p50 ms | p95 ms | p99 ms | Scaling efficiency |
|---------|-------|--------|--------|--------|--------------------|
| 1       |       |        |        |        | 1.00×              |
| 2       |       |        |        |        |                    |
| 3       |       |        |        |        |                    |
| 5       |       |        |        |        |                    |

## Fault Tolerance
| Scenario               | Recovery time | Data loss |
|------------------------|---------------|-----------|
| Reader crash+restart   |               | None      |
| Writer crash+restart   |               | None      |
| Coordinator failover   |               | None      |
```

### Phase 8 Verification Checklist
- [ ] Every request produces a structured JSON log line with `trace_id`.
- [ ] `benchmark_report.md` filled with real numbers.
- [ ] QPS at 5 readers is ≥ 3.5× QPS at 1 reader.
- [ ] Recall@10 ≥ 0.90 at SIFT1M scale.
- [ ] `pytest tests/` — 100% pass rate.

---

## Cross-Cutting Implementation Notes

### Threading model
- All FastAPI handlers are `async def`. All I/O (disk, network, etcd) uses `await`.
- CPU-heavy Faiss calls (`index.train`, `index.search`) must be offloaded:
  ```python
  loop = asyncio.get_event_loop()
  result = await loop.run_in_executor(None, index.search, query, top_k)
  ```
  Failure to do this will block the event loop and tank throughput.

### Memory layout
- Always store vectors as `np.float32` contiguous arrays (`np.ascontiguousarray`).
- For the MemTable brute-force search, rebuild `faiss.IndexFlatL2` from current vectors each time (acceptable for ≤ 50 K vectors; rebuilding takes < 50 ms).
- For large Segments loaded into Reader cache, keep the Faiss index in RAM; do not reload from disk per query.

### Concurrency safety on Writer
- Python's GIL does not protect `list.append`. Use `asyncio.Lock` around MemTable mutations:
  ```python
  async with self._memtable_lock:
      self._memtable.insert(ids, vectors)
  ```
- WAL file writes must hold a separate `asyncio.Lock` — do not interleave two concurrent inserts in the log.

### Segment ID namespace
- `segment_id = f"{shard_id:04d}-{snapshot_id:010d}-{uuid4().hex[:8]}"` gives sortable, debuggable IDs.

### Error handling
- All inter-node HTTP calls: timeout=5 s, retry=3, exponential backoff.
- If a Faiss index is empty (ntotal=0), skip search and return empty result — Faiss panics on search of empty index.
- Graceful shutdown: FastAPI lifespan context should flush MemTable, close WAL, deregister from coordinator.

---

## Quick Reference: Port Assignments

| Service | Port |
|---|---|
| etcd | 2379 |
| coordinator-0 | 8000 |
| coordinator-1 | 8001 |
| coordinator-2 | 8002 |
| writer-0 | 8100 |
| reader-0 | 8200 |
| reader-1 | 8201 |
| reader-2 | 8202 |
| reader-3 | 8203 |
| reader-4 | 8204 |

## Quick Reference: Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VECSCALE_NODE_ROLE` | `reader` | `coordinator`, `writer`, or `reader` |
| `VECSCALE_NODE_ID` | `node-0` | Unique node identifier |
| `VECSCALE_DIM` | `128` | Vector dimension |
| `VECSCALE_NLIST` | `256` | IVF_FLAT centroid count |
| `VECSCALE_NPROBE` | `32` | IVF_FLAT search probes |
| `VECSCALE_SEGMENT_FLUSH_THRESHOLD` | `50000` | Vectors before flush |
| `VECSCALE_SEGMENT_MERGE_THRESHOLD` | `4` | Segments before merge |
| `VECSCALE_SHARED_STORAGE_PATH` | `/shared_storage` | Shared storage mount |
| `VECSCALE_WAL_PATH` | `/wal` | WAL directory (writer only) |
| `VECSCALE_COORDINATOR_URLS` | `["http://coordinator-0:8000"]` | Coordinator list |
| `VECSCALE_ETCD_ENDPOINTS` | `["http://etcd:2379"]` | etcd cluster endpoints |
