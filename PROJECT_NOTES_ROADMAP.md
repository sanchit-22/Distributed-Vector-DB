# VecScaleDB Project Notes And Roadmap

This document explains the project in simple language. It is written for someone who is not already familiar with distributed databases.

## 1. What This Project Is

VecScaleDB is a small distributed vector database.

A normal database stores rows like:

```text
id = 1, name = "Priya", age = 21
```

A vector database stores IDs with long lists of numbers:

```text
id = 1, vector = [0.12, 0.98, 0.44, ...]
```

These vectors usually represent meaning. For example:

- an image can become a vector,
- a sentence can become a vector,
- a product can become a vector,
- a song can become a vector.

Searching a vector database means:

```text
Find the stored vectors most similar to this query vector.
```

This is useful for:

- image similarity search,
- recommendation systems,
- semantic search,
- AI memory,
- document retrieval,
- product matching.

## 2. Simple Example

Imagine every movie is converted into a vector.

If the query is:

```text
"space adventure with robots"
```

the system converts that query into numbers and searches for movies with nearby vectors.

Instead of matching exact words, it finds similar meaning.

## 3. Why Distributed?

One machine can only store and search so much data.

A distributed system splits work across multiple services:

- one service writes data,
- several services read/search data,
- another service coordinates the cluster,
- shared storage keeps persistent segment files,
- etcd stores cluster metadata.

This project demonstrates those ideas in a small local Docker setup.

## 4. Main Parts Of The System

### Writer

The writer receives new vectors.

Its job:

- accept inserts,
- write them safely to the WAL,
- keep recent data in memory,
- flush large batches to segment files,
- tell the coordinator about new segments.

Think of the writer as the intake desk.

### Reader

Readers answer search requests.

Their job:

- load segment files,
- keep indexes in memory,
- search vectors quickly,
- return the best matches.

Think of readers as search workers.

### Coordinator

The coordinator manages cluster information.

Its job:

- know which readers exist,
- know which writer exists,
- know which segment files exist,
- route search requests,
- handle leader failover.

Think of the coordinator as the traffic controller.

## Current Validation Status

As of the latest local validation:

- The Python automated suite passes: `66 passed`.
- Single-node mode supports health checks, insert, search, WAL recovery, and clean validation errors.
- Distributed Docker mode starts 10 services: 3 coordinators, 1 writer, 5 readers, and etcd.
- Writer insert through port `8100` creates durable WAL entries and flushed segments.
- Coordinator search through port `8000` finds newly inserted vectors through readers.
- All five readers report ready and loaded segments.
- Metrics endpoints are available on coordinator, writer, and reader services.
- Reader failure was tested by stopping one reader; searches continued without HTTP 5xx.
- Coordinator failover was tested by stopping `coordinator-0` and searching through `coordinator-1`.

This does not mean every possible production failure has been exhausted. It means
the project is working for the implemented Phase 0 through Phase 8 scope and has
passed the expected local, integration, and Docker smoke tests.

### WAL

WAL means Write-Ahead Log.

Before the writer says "insert completed", it writes the operation to disk.

This protects data if the writer crashes.

If the writer restarts, it replays the WAL and restores acknowledged writes.

### MemTable

The MemTable is an in-memory buffer.

New vectors first live here after they are written to the WAL.

When enough vectors collect, the MemTable is flushed to a segment.

### Segment

A segment is an immutable saved block of vectors.

Each segment has:

- `index.faiss`: search index,
- `ids.npy`: vector IDs,
- `vectors.npy`: raw vectors,
- `meta.json`: segment metadata.

Readers load segments and search them.

### etcd

etcd is a small metadata database used by distributed systems.

In this project, etcd stores:

- registered nodes,
- segment metadata,
- current snapshot,
- coordinator leader.

## 5. How Data Moves Through The System

### Insert Flow

1. User sends vectors to the writer.
2. Writer writes the operation to WAL.
3. Writer puts vectors in MemTable.
4. When MemTable reaches 50,000 vectors, writer creates a segment.
5. Writer saves the segment to shared storage.
6. Writer registers the segment with coordinator.
7. Readers discover the new segment and load it.

### Search Flow

1. User sends query vector to coordinator.
2. Coordinator chooses a reader.
3. Reader searches loaded segments.
4. Reader returns nearest vector IDs.
5. Coordinator returns final results to the user.

## 6. What Each Phase Built

### Phase 0: Project Skeleton

Created the basic Python package, Docker setup, config, models, and dataset scripts.

### Phase 1: Single-Node Search

Built a simple vector search server using Faiss.

It could:

- insert vectors,
- create segment files,
- search nearest vectors.

### Phase 2: Durability And Storage

Added safer storage.

It introduced:

- WAL,
- MemTable,
- LSM engine,
- crash recovery,
- snapshots,
- segment merging.

### Phase 3: Coordinator

Added the coordinator service.

It introduced:

- node registration,
- segment registration,
- etcd metadata,
- leader election,
- routing.

### Phase 4: Writer Service

Moved ingestion into a dedicated writer service.

The writer now handles:

- inserts,
- deletes,
- WAL,
- flushes,
- segment registration.

### Phase 5: Reader Service

Added reader services and distributed search.

Readers:

- load segments,
- search locally,
- return results to coordinator.

Coordinator:

- sends search requests to readers,
- merges results,
- handles partial failures.

### Phase 6: Scaling To Five Readers

Added five reader containers.

The coordinator can use replicated-reader mode:

- all readers load all segments,
- each query goes to one reader,
- QPS can improve because requests are spread out.

### Phase 7: Fault Tolerance

Made the system survive crashes better.

Verified:

- reader restart,
- writer crash and WAL recovery,
- coordinator failover.

### Phase 8: Observability

Added logs and metrics.

Now each service exposes:

- JSON request logs,
- trace IDs,
- `/metrics`,
- request counts,
- request durations,
- service health gauges.

## 7. Current Project Status

Implemented and tested:

- single-node API,
- WAL durability,
- segment storage,
- snapshot-aware search,
- coordinator metadata,
- writer service,
- reader service,
- five-reader cluster,
- QPS benchmark script,
- fault tolerance checks,
- structured logs and metrics.

Latest full test result:

```text
62 passed
```

Latest live end-to-end result:

- writer inserted vectors,
- writer flushed a new segment,
- all 5 readers loaded it,
- coordinator search returned exact ID `700000`,
- distance was `0.0`,
- trace ID worked,
- metrics worked,
- reader restart recovered successfully.

## 8. What Is Not Fully Completed Yet

The code is complete through Phase 8, but full real benchmark numbers still need the SIFT1M dataset file.

Missing local file:

```text
data/sift1m/sift1m.hdf5
```

Once downloaded, the final benchmark report can be filled with real SIFT1M recall and QPS numbers.

## 9. Roadmap From Here

### Next Short-Term Work

1. Download SIFT1M.
2. Load all 1,000,000 vectors.
3. Run recall benchmark.
4. Run 1-reader QPS benchmark.
5. Run 2-reader QPS benchmark.
6. Run 3-reader QPS benchmark.
7. Run 5-reader QPS benchmark.
8. Fill `benchmark_report.md` with final numbers.

### Possible Future Improvements

1. Partition segments across readers instead of full replication.
2. Add multiple writer nodes.
3. Add a real object store such as S3 or MinIO.
4. Add Prometheus and Grafana dashboards.
5. Add authentication.
6. Add delete compaction for old persisted segments.
7. Add a client SDK.
8. Add web UI for inserts/search/health.

## 10. How To Explain This Project In A Presentation

Simple version:

```text
This project is a mini version of a distributed vector database like Milvus.
It stores AI-style vectors, writes them safely using a WAL, saves searchable
segments, distributes search across reader nodes, coordinates metadata through
etcd, survives node restarts, and exposes logs/metrics for monitoring.
```

One-minute explanation:

```text
VecScaleDB accepts high-dimensional vectors and lets users search for the nearest
vectors. It uses a writer node for ingestion, reader nodes for search, and a
coordinator for routing and metadata. Data is protected with a write-ahead log,
flushed into immutable segments, loaded by readers, and searched using Faiss.
The project also includes Docker orchestration, fault tolerance checks, benchmark
scripts, and observability through structured logs and metrics.
```

## 11. Important Files

```text
vecscaledb/writer/app.py          Writer API
vecscaledb/reader/app.py          Reader API
vecscaledb/coordinator/app.py     Coordinator API
vecscaledb/storage/wal.py         Write-ahead log
vecscaledb/storage/lsm.py         Storage engine
vecscaledb/index/segment.py       Segment files
vecscaledb/index/ivf_flat.py      Faiss index wrapper
vecscaledb/observability.py       Logs and metrics
docker-compose.yml                Full cluster
benchmark_report.md               Benchmark report
```

## 12. Glossary

Vector:
A list of numbers that represents meaning.

Nearest neighbor search:
Finding the vectors most similar to a query vector.

Faiss:
A library from Meta/Facebook for fast vector search.

WAL:
Write-Ahead Log. A safety file used to recover writes after crashes.

Segment:
A saved group of vectors and its search index.

MemTable:
Temporary in-memory storage before vectors become a segment.

Coordinator:
Service that manages routing and metadata.

Reader:
Service that searches vectors.

Writer:
Service that accepts and stores new vectors.

etcd:
Metadata store used for distributed coordination.

QPS:
Queries per second. Measures how many searches the system can answer.

Recall:
Measures how many correct nearest neighbors the search returns.
