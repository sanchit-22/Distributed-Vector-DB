# VecScaleDB

VecScaleDB is a distributed vector database built for a distributed systems
project. It stores integer IDs with fixed-size vector embeddings, persists
writes through a WAL and immutable Faiss-backed segments, and supports
distributed search through coordinator, writer, reader, shared storage, and etcd
services.

This README is the main runbook for setting up, running, testing, and evaluating
the project.

## 1. Prerequisites

Install:

- Python 3.11 or newer
- Docker and Docker Compose
- Bash
- Git

For benchmark data:

- Enough disk space for SIFT1M
- Network access to download the dataset

## 2. Project Setup

From the project root:

```bash
cd /home/san22chit/Documents/IIITH/Sem4/DistributedSystems/VectorDB/Distributed-Vector-DB
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you already have the environment:

```bash
source venv/bin/activate
```

On Windows PowerShell, the equivalent activation command is:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Run Tests

Run the automated test suite:

```bash
python -m pytest tests -v
```

Expected project result from the current notes:

```text
66 passed
```

## 4. Runtime Options

The project can run in two modes:

| Mode | Command | Use case |
| --- | --- | --- |
| Single-node | `python -m vecscaledb.node` | Simple local insert/search/WAL demo |
| Distributed Docker cluster | `docker compose up -d --build` | Main project mode with coordinators, writer, readers, and etcd |

Important port difference:

```text
single-node mode:
  insert/search/health -> port 8000

distributed mode:
  coordinator search   -> port 8000
  writer insert/delete -> port 8100
  readers              -> ports 8200-8204
```

## 5. Run Single-Node Mode

Single-node mode runs one process that handles insert, search, WAL recovery, and
segment storage.

Start:

```bash
python -m vecscaledb.node
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Stop with:

```text
Ctrl+C
```

Use single-node mode for:

- quick local API testing,
- WAL recovery demonstration,
- `vecscaledb.bench.recall_eval`, because that script expects `/insert` and
  `/search` on the same server.

## 6. Run Distributed Cluster

Before starting Docker mode, stop the single-node server if it is running,
because both single-node and coordinator-0 use port `8000`.

Start the full cluster:

```bash
docker compose down
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

This starts:

```text
etcd
coordinator-0  http://127.0.0.1:8000
coordinator-1  http://127.0.0.1:8001
coordinator-2  http://127.0.0.1:8002
writer-0       http://127.0.0.1:8100
reader-0       http://127.0.0.1:8200
reader-1       http://127.0.0.1:8201
reader-2       http://127.0.0.1:8202
reader-3       http://127.0.0.1:8203
reader-4       http://127.0.0.1:8204
```

Check containers:

```bash
docker compose ps
```

Check health:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8200/ready
curl http://127.0.0.1:8201/ready
curl http://127.0.0.1:8202/ready
curl http://127.0.0.1:8203/ready
curl http://127.0.0.1:8204/ready
```

Stop the cluster:

```bash
docker compose down
```

## 7. Quick Distributed Demo

In distributed mode, readers search flushed segment files. They do not directly
search the writer's live MemTable. For a tiny demo, set the flush threshold to
`2` so two inserted vectors immediately become a segment.

Start demo cluster:

```bash
docker compose down
VECSCALE_SEGMENT_FLUSH_THRESHOLD=2 docker compose up -d --build
```

Insert two 128-dimensional vectors into the writer:

```bash
python - <<'PY'
import httpx

dim = 128
v1 = [1.0] + [0.0] * (dim - 1)
v2 = [0.0, 1.0] + [0.0] * (dim - 2)

body = {
    "ids": [101, 102],
    "vectors": [v1, v2],
}

res = httpx.post("http://127.0.0.1:8100/insert", json=body)
print(res.status_code)
print(res.json())
PY
```

Search through the coordinator:

```bash
python - <<'PY'
import httpx

dim = 128
query = [1.0] + [0.0] * (dim - 1)

body = {
    "query": query,
    "top_k": 5,
    "nprobe": 32,
}

res = httpx.post("http://127.0.0.1:8000/search", json=body)
print(res.status_code)
print(res.json())
PY
```

Expected result:

```text
ID 101 should appear with distance 0.0
```

For benchmark loading, do not keep the threshold at `2`. Restart with the
default threshold:

```bash
docker compose down
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

## 8. Interactive Client

The project includes an interactive client:

```bash
python -m vecscaledb.interactive_client
```

Supported operations:

```text
insert
search
delete
health
segments
ready
metrics
quit
```

Common ports:

```text
insert/delete -> 8100 writer
search        -> 8000 coordinator
ready         -> 8200 reader
single-node   -> 8000
```

Example:

```text
Operation [search]: insert
Host [127.0.0.1]:
Port [8100]:
ID: 101
Vector: 1,0,0,0,...
ID:
```

## 9. Inspect Stored Data

Writer health shows the writer/storage view:

```bash
curl http://127.0.0.1:8100/health
```

Look for:

```text
ntotal
memtable_size
segments
wal_lsn
```

Coordinator segments show persisted and registered segment data:

```bash
curl http://127.0.0.1:8000/segments
```

Sum persisted vectors:

```bash
python - <<'PY'
import httpx

segments = httpx.get("http://127.0.0.1:8000/segments").json()
print("segments:", len(segments))
print("vectors in persisted segments:", sum(s["num_vectors"] for s in segments))
PY
```

Note: vectors still in the writer MemTable appear in writer `/health`, but they
do not appear in `/segments` until flushed.

## 10. Download And Inspect SIFT Dataset

Download SIFT1M:

```bash
bash data/download_sift.sh
```

Expected dataset:

```text
data/sift1m/sift1m.hdf5
```

Inspect dataset shapes:

```bash
python - <<'PY'
import h5py

path = "data/sift1m/sift1m.hdf5"

with h5py.File(path, "r") as f:
    for key in f.keys():
        print(key, f[key].shape, f[key].dtype)
PY
```

Expected main arrays:

```text
train      (1000000, 128)
test       (10000, 128)
neighbors  (10000, 100)
```

## 11. Load Benchmark Data

For benchmark loading, use the normal flush threshold, not `2`:

```bash
docker compose down
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

Load a small subset:

```bash
python -m vecscaledb.bench.load_dataset \
  --dataset data/sift1m/sift1m.hdf5 \
  --writer-url http://127.0.0.1:8100 \
  --batch-size 10000 \
  --limit 100000
```

Load full SIFT1M:

```bash
python -m vecscaledb.bench.load_dataset \
  --dataset data/sift1m/sift1m.hdf5 \
  --writer-url http://127.0.0.1:8100 \
  --batch-size 10000
```

Confirm data is loaded:

```bash
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8000/segments
```

## 12. Run QPS Benchmark

Run search throughput benchmark against the coordinator:

```bash
python -m vecscaledb.bench.benchmark_qps \
  --dataset data/sift1m/sift1m.hdf5 \
  --coordinator-url http://127.0.0.1:8000 \
  --query-limit 1000 \
  --top-k 10 \
  --nprobe 32 \
  --concurrency 32 \
  --duration 60
```

This reports:

```text
total queries
errors
QPS
p50 latency
p95 latency
p99 latency
```

## 13. Run QPS-vs-Readers Scaling Evaluation

This is the main Figure 10b-style distributed scalability benchmark.

```bash
python -m vecscaledb.bench.scaling_eval \
  --dataset data/sift1m/sift1m.hdf5 \
  --coordinator-url http://127.0.0.1:8000 \
  --reader-counts 1 2 3 4 5 \
  --duration 60 \
  --concurrency 32 \
  --query-limit 1000 \
  --output scaling_results.json \
  --auto-scale
```

What it does:

- reads query vectors from SIFT1M,
- starts/stops reader containers automatically,
- benchmarks with 1, 2, 3, 4, and 5 readers,
- sends concurrent search requests to the coordinator,
- records QPS and latency,
- writes results to `scaling_results.json`.

## 14. Run Recall Evaluation

The current `recall_eval` script is single-node oriented because it inserts and
searches through the same host/port.

Run it like this:

```bash
docker compose down
python -m vecscaledb.node
```

In another terminal:

```bash
source venv/bin/activate
python -m vecscaledb.bench.recall_eval \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset data/sift1m/sift1m.hdf5
```

Do not run `recall_eval` against distributed coordinator port `8000`, because
the coordinator does not expose `/insert`. In distributed mode, insert goes to
writer port `8100`.

## 15. Scatter-Gather Shard Mode

The default Docker compose file uses replicated readers:

```text
every reader loads all shards
coordinator load-balances queries round-robin
```

To test scatter-gather sharding:

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.scatter.yml up -d --build
```

In this mode:

- writer uses 5 shards,
- IDs are assigned by `id % num_shards`,
- each reader loads only its own shard,
- coordinator fans each query to all readers,
- coordinator merges partial results.

## 16. Fault Tolerance Checks

Reader restart:

```bash
docker compose stop reader-3
docker compose start reader-3
sleep 15
curl http://127.0.0.1:8203/ready
```

Coordinator failover:

```bash
docker compose stop coordinator-0
sleep 15
curl http://127.0.0.1:8001/health
```

Search through coordinator-1:

```bash
python - <<'PY'
import httpx

query = [1.0] + [0.0] * 127
body = {"query": query, "top_k": 5, "nprobe": 32}
print(httpx.post("http://127.0.0.1:8001/search", json=body).json())
PY
```

Start coordinator-0 again:

```bash
docker compose start coordinator-0
```

Writer restart:

```bash
docker compose stop writer-0
docker compose start writer-0
sleep 10
curl http://127.0.0.1:8100/health
```

## 17. Metrics And Logs

Metrics:

```bash
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8100/metrics
curl http://127.0.0.1:8200/metrics
```

Logs:

```bash
docker compose logs --tail=50 coordinator-0
docker compose logs --tail=50 writer-0
docker compose logs --tail=50 reader-0
```

Follow all logs:

```bash
docker compose logs -f
```

Search with a trace ID:

```bash
python - <<'PY'
import httpx

query = [1.0] + [0.0] * 127
body = {"query": query, "top_k": 5, "nprobe": 32}
headers = {"x-trace-id": "manual-readme-test"}
print(httpx.post("http://127.0.0.1:8000/search", json=body, headers=headers).json())
PY
```

Find it in logs:

```bash
docker compose logs coordinator-0 | grep manual-readme-test
```

## 18. Common Troubleshooting

### `ModuleNotFoundError: No module named 'httpx'`

Activate the project environment and install dependencies:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

### `404 Not Found` for `http://127.0.0.1:8000/insert`

You are probably running distributed mode. In distributed mode:

```text
insert -> writer port 8100
search -> coordinator port 8000
```

Use:

```text
http://127.0.0.1:8100/insert
```

### Inserted small vectors are not found in distributed search

They may still be in the writer MemTable. Distributed readers search flushed
segments. For quick demos, restart with:

```bash
docker compose down
VECSCALE_SEGMENT_FLUSH_THRESHOLD=2 docker compose up -d --build
```

For benchmarks, return to the default threshold:

```bash
docker compose down
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

### Port `8000` already in use

Stop single-node mode or old Docker containers:

```bash
docker compose down
```

If a local Python server is running, stop it with `Ctrl+C`.
