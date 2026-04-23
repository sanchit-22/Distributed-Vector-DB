# VecScaleDB Phase 0/1 Guide

This document explains:

- what each current file/module does,
- the implemented architecture for Phase 0 and Phase 1,
- how to run and test each module.

Scope of this guide is the current repository state under `Distributed-Vector-DB`.

## 1. High-Level Architecture (Implemented So Far)

### Phase 0 (Skeleton + Environment)

- Packaging and dependency setup is in place (`pyproject.toml`, `requirements.txt`).
- Core config and schema contracts exist (`vecscaledb/config.py`, `vecscaledb/models.py`).
- Docker skeleton exists for future distributed roles (`docker-compose.yml`, `Dockerfile`).
- Dataset tooling exists for SIFT1M/SIFT10K (`data/download_sift.sh`, `vecscaledb/bench/load_dataset.py`).

### Phase 1 (Working Single Node)

- A fully working FastAPI single-node server exists in `vecscaledb/node.py`.
- In-memory writes go to a MemTable, and are flushed to immutable on-disk segments.
- Each segment contains a persisted Faiss IVF_FLAT index via:
  - `vecscaledb/index/ivf_flat.py`
  - `vecscaledb/index/segment.py`
- Search fan-ins results from:
  - persisted segments, and
  - current MemTable
    then merges/deduplicates top-k results.
- Recall evaluation script exists (`vecscaledb/bench/recall_eval.py`).
- Unit and integration tests exist under `tests/`.

## 2. Current Data Flow

### Insert Path

1. `POST /insert` (`vecscaledb/node.py`) receives `InsertRequest`.
2. Vectors and IDs are buffered in `_memtable`.
3. When memtable size crosses `segment_flush_threshold`, `_flush()` is triggered.
4. `_flush()` builds a segment using `Segment.create(...)`.
5. `Segment.create(...)` trains/adds IVF_FLAT (`IVFFlatIndex`) and persists files:
   - `index.faiss`
   - `ids.npy`
   - `meta.json`

### Search Path

1. `POST /search` receives `SearchRequest`.
2. Server searches all loaded segments (`Segment.search(...)`).
3. Server brute-force searches current memtable (`_memtable_search`).
4. `_merge_results(...)` deduplicates by ID and keeps smallest distance.
5. Returns `SearchResult` with top-k IDs/distances.

## 3. File-by-File Reference

### Root files

- `pyproject.toml`
  - Build system: Hatchling.
  - Package metadata for `vecscaledb`.
  - Pytest option `asyncio_mode = auto`.

- `requirements.txt`
  - Runtime + test dependencies: FastAPI, uvicorn, httpx, faiss-cpu, numpy, pydantic, pydantic-settings, structlog, pytest, pytest-asyncio, h5py, tqdm.

- `docker-compose.yml`
  - Defines services: `etcd`, `coordinator-0`, `writer-0`, `reader-0`.
  - Additional readers are present as commented templates.
  - Note: service commands reference `vecscaledb.coordinator.app`, `vecscaledb.writer.app`, `vecscaledb.reader.app` which are placeholders for later phases.

- `Dockerfile`
  - Python 3.11 slim image.
  - Installs `requirements.txt`, copies project, sets `PYTHONPATH=/app`.

- `agent.md`
  - Project implementation plan/spec reference.

### Data tooling

- `data/download_sift.sh`
  - Downloads SIFT1M HDF5 to `data/sift1m/sift1m.hdf5`.
  - SIFT10K is not separately downloaded; it is sliced from SIFT1M by loader code.

### Package

- `vecscaledb/__init__.py`
  - Exposes package version: `__version__ = "0.1.0"`.

- `vecscaledb/config.py`
  - `Settings` class using Pydantic settings with `VECSCALE_` env prefix.
  - Central tunables: dimensions, IVF params (`nlist`, `nprobe`), flush/merge thresholds, storage paths, node role info.

- `vecscaledb/models.py`
  - Shared Pydantic models:
    - `InsertRequest`
    - `SearchRequest`
    - `SearchResult`
    - `SegmentMeta`
    - `NodeInfo`

- `vecscaledb/node.py`
  - Single-node FastAPI app for Phase 1.
  - Endpoints:
    - `POST /insert`
    - `POST /search`
    - `GET /health`
  - Maintains in-process MemTable and loaded segment list.
  - Loads existing segments on startup and flushes remaining memtable on shutdown.

### Index layer

- `vecscaledb/index/ivf_flat.py`
  - `IVFFlatIndex` wrapper over Faiss `IndexIVFFlat` + `IndexIDMap2`.
  - Public lifecycle methods:
    - `train(vectors)`
    - `add(vectors, ids)`
    - `search(query, top_k, nprobe)`
    - `save(path)`
    - `load(path)`
    - `ntotal` property

- `vecscaledb/index/segment.py`
  - Immutable persisted segment abstraction.
  - `Segment.create(...)` builds index and writes segment directory.
  - `Segment.load(path)` restores from disk.
  - `search(...)` delegates to underlying index.

- `vecscaledb/index/__init__.py`
  - Package marker for index module.

### Bench utilities

- `vecscaledb/bench/load_dataset.py`
  - `load_sift1m(path)` returns train/test/ground-truth arrays from HDF5.
  - `load_sift10k(path)` returns 10K train + 100 query subset for fast iteration.

- `vecscaledb/bench/recall_eval.py`
  - End-to-end recall benchmark against a running server.
  - Inserts SIFT10K over REST, queries top-100, computes Recall@10 and Recall@100.
  - Exits with failure if Recall@10 < 0.95.

- `vecscaledb/bench/__init__.py`
  - Package marker for benchmarking module.

### Distributed-role placeholders (future phases)

- `vecscaledb/coordinator/__init__.py`
- `vecscaledb/writer/__init__.py`
- `vecscaledb/reader/__init__.py`
- `vecscaledb/storage/__init__.py`
  - Currently placeholders only; logic will be added in later phases.

### Tests

- `tests/unit/test_ivf_flat.py`
  - Validates IVF wrapper behavior:
    - train/add/search
    - add-before-train error path
    - ID validity
    - save/load consistency

- `tests/unit/test_segment.py`
  - Validates segment behavior:
    - file creation on `Segment.create`
    - successful `Segment.load`
    - search consistency before/after reload
    - metadata correctness

- `tests/integration/test_single_node.py`
  - In-process integration test using FastAPI `TestClient`.
  - Covers `/health`, insert path, search behavior, and flush trigger behavior.

- `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
  - Package markers for tests.

## 4. Environment Variables (Most Used)

All prefixed with `VECSCALE_`:

- `VECSCALE_DIM` (default `128`)
- `VECSCALE_NLIST` (default `256`)
- `VECSCALE_NPROBE` (default `32`)
- `VECSCALE_SEGMENT_FLUSH_THRESHOLD` (default `50000`)
- `VECSCALE_SHARED_STORAGE_PATH` (default `/shared_storage`)
- `VECSCALE_WAL_PATH` (default `/wal`)
- `VECSCALE_NODE_ROLE` / `VECSCALE_NODE_ID`

## 5. How To Run (Phase 0/1)

Run all commands from project root:
`/home/san22chit/Documents/IIITH/Sem4/DistributedSystems/VectorDB/Distributed-Vector-DB`

### 5.1 Install

```bash
python -m pip install -e .
```

### 5.2 Quick Smoke Checks

```bash
python -c "import vecscaledb; print(vecscaledb.__version__)"
docker compose config
pytest tests/ --collect-only
```

### 5.3 Download Dataset

```bash
chmod +x data/download_sift.sh
./data/download_sift.sh
```

### 5.4 Run Single-Node API Server (Phase 1)

```bash
python -m vecscaledb.node
```

Server starts on `http://0.0.0.0:8000`.

### 5.5 Run Recall Evaluation Against Live Server

In another terminal:

```bash
python -m vecscaledb.bench.recall_eval --host 127.0.0.1 --port 8000 --dataset data/sift1m/sift1m.hdf5
```

Expected: `Recall@10 >= 0.95`.

## 6. How To Test Each Module

### 6.1 Unit Tests: Index + Segment

```bash
pytest tests/unit/test_ivf_flat.py tests/unit/test_segment.py -v
```

### 6.2 Integration Test: Single Node API

```bash
pytest tests/integration/test_single_node.py -v
```

### 6.3 Full Test Suite

```bash
pytest tests/ -v
```

## 7. Notes and Known Boundaries

- `docker-compose.yml` is a distributed skeleton for future phases; coordinator/writer/reader app modules are not yet implemented.
- The Phase 1 functional runtime is `vecscaledb.node` (single process).
- SIFT10K in this repo is generated as a slice via loader code, not as an independent source file.
