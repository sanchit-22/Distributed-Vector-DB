# VecScaleDB

VecScaleDB is a distributed vector database built for the Distributed Systems
project. It stores integer IDs with fixed-size vector embeddings, persists
writes through a WAL and immutable Faiss-backed segments, and serves distributed
search through coordinator, writer, reader, shared storage, and etcd services.

This README is the main runbook for setup, local demos, dataset loading,
benchmarking, and report generation.

Repository: https://github.com/sanchit-22/Distributed-Vector-DB

## 1. Prerequisites

Install:

- Python 3.11 or newer
- Docker and Docker Compose
- Bash
- Git
- `wget` for dataset download scripts

For benchmark data:

- SIFT1M needs roughly 500 MB.
- Deep1M uses the public `deep-image-96-angular.hdf5` file and needs several GB.
- Network access is required for the download scripts.

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

If the environment already exists:

```bash
source venv/bin/activate
```

## 3. Verify The Codebase

Run the full test suite:

```bash
python -m pytest tests -q
```

Expected result for the current codebase:

```text
71 passed, 2 warnings
```

Run the proposal feature suite:

```bash
python -m pytest tests/feature -q
```

Expected result:

```text
17 passed, 2 warnings
```

Check the main benchmark/report entry points:

```bash
python -m vecscaledb.bench.load_dataset --help
python -m vecscaledb.bench.benchmark_qps --help
python -m vecscaledb.bench.scaling_eval --help
python report/run_dataset_metrics.py --help
bash -n data/download_sift.sh report/download_deep1m.sh report/run_full_benchmark.sh
docker compose config --quiet
docker compose -f docker-compose.yml -f docker-compose.scatter.yml config --quiet
```

## 4. Runtime Modes

| Mode | Command | Use case |
| --- | --- | --- |
| Single-node | `python -m vecscaledb.node` | Simple local insert/search/WAL demo |
| Distributed Docker cluster | `docker compose up -d --build` | Main project mode with coordinators, writer, readers, and etcd |

Port map:

```text
single-node mode:
  insert/search/health -> 8000

distributed mode:
  coordinator search   -> 8000, 8001, 8002
  writer insert/delete -> 8100
  readers              -> 8200, 8201, 8202, 8203, 8204
```

## 5. Single-Node Mode

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

Stop with `Ctrl+C`.

Use this mode for quick API tests or the legacy single-node recall script:

```bash
python -m vecscaledb.bench.recall_eval \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset data/sift1m/sift1m.hdf5
```

Do not run `recall_eval` against distributed coordinator port `8000`; in
distributed mode inserts go to writer port `8100`, while coordinator port `8000`
does not expose `/insert`.

## 6. Distributed Cluster

Stop single-node mode before starting Docker because both use port `8000`.

Start the normal 128D SIFT-compatible cluster:

```bash
docker compose down
unset VECSCALE_DIM
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
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
```

Stop:

```bash
docker compose down
```

Important dimension rule:

- SIFT10K and SIFT1M are 128D, so use the default `VECSCALE_DIM=128`.
- Deep1M is 96D, so the cluster must be started with `VECSCALE_DIM=96`.
- Switching between 128D and 96D requires a clean `shared_storage/` and `wal/`.

Clean runtime state manually when switching datasets or dimensions:

```bash
docker compose down
mkdir -p shared_storage wal
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
```

The full benchmark script in section 14 runs this cleanup automatically.

## 7. Quick Distributed Demo

Readers search flushed segment files. They do not directly search the writer's
live MemTable. For a tiny demo, set the flush threshold to `2` so two inserted
vectors immediately become a segment.

Start the demo cluster:

```bash
docker compose down
VECSCALE_SEGMENT_FLUSH_THRESHOLD=2 docker compose up -d --build
```

Insert two 128D vectors into the writer:

```bash
python - <<'PY'
import httpx

dim = 128
body = {
    "ids": [101, 102],
    "vectors": [
        [1.0] + [0.0] * (dim - 1),
        [0.0, 1.0] + [0.0] * (dim - 2),
    ],
}

response = httpx.post("http://127.0.0.1:8100/insert", json=body)
print(response.status_code)
print(response.json())
PY
```

Search through the coordinator:

```bash
python - <<'PY'
import httpx

dim = 128
body = {
    "query": [1.0] + [0.0] * (dim - 1),
    "top_k": 5,
    "nprobe": 32,
}

response = httpx.post("http://127.0.0.1:8000/search", json=body)
print(response.status_code)
print(response.json())
PY
```

Expected result: ID `101` should appear with distance `0.0`.

Return to benchmark settings after the tiny demo:

```bash
docker compose down
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

## 8. Interactive Client

Run:

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

## 9. Inspect Stored Data

Writer health shows MemTable, WAL, and local storage state:

```bash
curl http://127.0.0.1:8100/health
```

Coordinator segment metadata shows flushed, persisted segments:

```bash
curl http://127.0.0.1:8000/segments
```

Count vectors in persisted segments:

```bash
python - <<'PY'
import httpx

segments = httpx.get("http://127.0.0.1:8000/segments").json()
print("segments:", len(segments))
print("vectors in persisted segments:", sum(s["num_vectors"] for s in segments))
PY
```

Vectors still in the writer MemTable appear in writer `/health`, but they do
not appear in `/segments` until flushed.

## 10. Download Datasets

### SIFT1M And SIFT10K

Download SIFT1M:

```bash
bash data/download_sift.sh
```

Expected file:

```text
data/sift1m/sift1m.hdf5
```

SIFT10K is not a separate download. The benchmark tooling treats SIFT10K as
the first 10,000 train vectors from SIFT1M.

Inspect SIFT arrays:

```bash
python - <<'PY'
import h5py

path = "data/sift1m/sift1m.hdf5"
with h5py.File(path, "r") as h5:
    for key in h5.keys():
        print(key, h5[key].shape, h5[key].dtype)
PY
```

Expected main arrays:

```text
train      (1000000, 128)
test       (10000, 128)
neighbors  (10000, 100)
```

### Deep1M

Download Deep1M:

```bash
bash report/download_deep1m.sh
```

Expected file:

```text
data/deep1m/deep1m.hdf5
```

The public file is named `deep-image-96-angular.hdf5`. In this project,
Deep1M means the first 1,000,000 train vectors from that file.

Inspect Deep arrays:

```bash
python - <<'PY'
import h5py

path = "data/deep1m/deep1m.hdf5"
with h5py.File(path, "r") as h5:
    for key in h5.keys():
        print(key, h5[key].shape, h5[key].dtype)
PY
```

Deep vectors are 96D. Its public ground truth is angular/cosine-style, while
VecScaleDB currently searches with L2. Therefore Deep1M is valid for
load/QPS/scaling measurements, but L2 recall is skipped by default unless you
force it or recompute L2 ground truth.

## 11. Load Benchmark Data Manually

Start a clean 128D cluster for SIFT:

```bash
docker compose down
mkdir -p shared_storage wal
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
unset VECSCALE_DIM
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

Load a SIFT subset:

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

For Deep1M, start a clean 96D cluster first:

```bash
docker compose down
mkdir -p shared_storage wal
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
VECSCALE_DIM=96 VECSCALE_SEGMENT_FLUSH_THRESHOLD=50000 docker compose up -d --build
```

Then load Deep1M through the dataset metrics runner:

```bash
python report/run_dataset_metrics.py \
  --dataset deep1m \
  --load \
  --train-limit 1000000 \
  --query-limit 1000 \
  --output report/results/deep1m_metrics.json
```

Confirm data is loaded:

```bash
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8000/segments
```

## 12. QPS Benchmark

Run SIFT search throughput against the coordinator:

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

Current output fields:

```text
total_queries       successful complete searches
attempts            total request attempts
errors              failed or incomplete attempts
error_rate          errors / attempts
qps                 successful complete searches per second
p50/p95/p99         latency percentiles for successful searches only
```

So `errors` can be greater than `total_queries`; they measure failed attempts
beside successful attempts, not errors inside the successful query count.

## 13. QPS-vs-Readers Scaling

This is the Figure 10b-style QPS-vs-reader-count benchmark.

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

- reads query vectors from the HDF5 dataset,
- starts/stops reader containers when `--auto-scale` is set,
- waits for reader `/ready`,
- waits for the coordinator registry to match the active readers,
- sends concurrent search requests to the coordinator,
- writes QPS, latency, attempts, errors, and error rate to JSON.

On a single laptop, QPS may flatten or drop as readers increase because all
reader containers compete for the same CPU, memory bandwidth, and disk cache.
That is different from a real multi-machine cluster.

## 14. Dataset Metrics And Report Workflow

The recommended final workflow is `report/run_full_benchmark.sh`. It clears
runtime state between datasets, handles Docker root-owned files, starts the
correct 128D or 96D cluster, loads data, runs QPS/recall/scaling where
applicable, and regenerates the report PDF.

Run SIFT10K and SIFT1M:

```bash
bash report/run_full_benchmark.sh
```

Run SIFT10K, SIFT1M, and Deep1M:

```bash
bash report/download_deep1m.sh
RUN_DEEP=1 bash report/run_full_benchmark.sh
```

For a faster smoke run:

```bash
DURATION_SECONDS=10 CONCURRENCY=8 RECALL_QUERIES=100 bash report/run_full_benchmark.sh
```

For a faster Deep smoke run:

```bash
DURATION_SECONDS=10 CONCURRENCY=8 RECALL_QUERIES=100 RUN_DEEP=1 bash report/run_full_benchmark.sh
```

Generated metric files:

```text
report/results/sift10k_metrics.json
report/results/sift1m_metrics.json
report/results/deep1m_metrics.json
```

Generated report:

```text
report/vecscaledb_ann_benchmark_report.pdf
```

Regenerate the PDF from existing JSON results without rerunning benchmarks:

```bash
bash report/run_report.sh
```

The editable LaTeX report is:

```text
report/latex/vecscaledb_report.tex
```

Compile it manually if needed:

```bash
cd report/latex
pdflatex vecscaledb_report.tex
pdflatex vecscaledb_report.tex
```

## 15. One Dataset At A Time

SIFT10K smoke metrics:

```bash
docker compose down
mkdir -p shared_storage wal
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
VECSCALE_DIM=128 VECSCALE_SEGMENT_FLUSH_THRESHOLD=10000 docker compose up -d --build

python report/run_dataset_metrics.py \
  --dataset sift10k \
  --load \
  --qps \
  --recall \
  --duration 30 \
  --concurrency 16 \
  --query-limit 100 \
  --output report/results/sift10k_metrics.json
```

SIFT1M full metrics with scaling:

```bash
docker compose down
mkdir -p shared_storage wal
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
VECSCALE_DIM=128 VECSCALE_SEGMENT_FLUSH_THRESHOLD=50000 docker compose up -d --build

python report/run_dataset_metrics.py \
  --dataset sift1m \
  --load \
  --qps \
  --scaling \
  --auto-scale \
  --recall \
  --recall-queries 1000 \
  --recall-k 10 100 \
  --duration 60 \
  --concurrency 32 \
  --query-limit 1000 \
  --output report/results/sift1m_metrics.json
```

Deep1M full metrics with scaling:

```bash
bash report/download_deep1m.sh
docker compose down
mkdir -p shared_storage wal
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
VECSCALE_DIM=96 VECSCALE_SEGMENT_FLUSH_THRESHOLD=50000 docker compose up -d --build

python report/run_dataset_metrics.py \
  --dataset deep1m \
  --load \
  --qps \
  --scaling \
  --auto-scale \
  --recall \
  --duration 60 \
  --concurrency 32 \
  --query-limit 1000 \
  --output report/results/deep1m_metrics.json
```

Deep1M recall will be recorded as skipped because its ground truth metric is
angular while this VecScaleDB run uses L2.

## 16. Scatter-Gather Shard Mode

The default compose file uses replicated benchmark mode:

```text
each reader loads all shards
coordinator load-balances queries round-robin
```

To test scatter-gather sharding:

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.scatter.yml up -d --build
```

In scatter mode:

- writer uses 5 shards,
- IDs are assigned by `id % num_shards`,
- each reader loads only its own shard,
- coordinator fans each query to all readers,
- coordinator merges partial results.

## 17. Fault Tolerance Checks

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

## 18. Metrics And Logs

Prometheus-style metrics:

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

Find the trace:

```bash
docker compose logs coordinator-0 | grep manual-readme-test
```

## 19. Common Troubleshooting

### `ModuleNotFoundError: No module named 'httpx'`

Activate the project environment and install dependencies:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

### `404 Not Found` for `http://127.0.0.1:8000/insert`

You are running distributed mode. In distributed mode:

```text
insert/delete -> http://127.0.0.1:8100
search        -> http://127.0.0.1:8000
```

Use writer port `8100` for inserts.

### Inserted small vectors are not found in distributed search

They may still be in the writer MemTable. Distributed readers search flushed
segments. For quick demos, use:

```bash
docker compose down
VECSCALE_SEGMENT_FLUSH_THRESHOLD=2 docker compose up -d --build
```

For benchmarks, return to a larger threshold:

```bash
docker compose down
unset VECSCALE_SEGMENT_FLUSH_THRESHOLD
docker compose up -d --build
```

### Deep1M search returns dimension errors

Deep1M is 96D. Restart with a clean 96D cluster:

```bash
docker compose down
VECSCALE_DIM=96 VECSCALE_SEGMENT_FLUSH_THRESHOLD=50000 docker compose up -d --build
```

If old 128D segment files are still present, run the full benchmark script,
which clears state safely:

```bash
RUN_DEEP=1 bash report/run_full_benchmark.sh
```

### `Permission denied` while deleting `shared_storage/segments`

Docker may create root-owned files. The recommended full benchmark script
handles this automatically:

```bash
bash report/run_full_benchmark.sh
```

For manual cleanup, stop the cluster and use a Docker root helper:

```bash
docker compose down
docker compose run --rm --no-deps --user root writer-0 \
  sh -c 'rm -rf /shared_storage/* /wal/*'
```

### High error rate in QPS benchmarks

Check the new fields:

```text
attempts
errors
error_rate
incomplete_errors
request_errors
```

Then inspect coordinator logs:

```bash
docker compose logs coordinator-0 | grep reader.search.failed
```

If errors are mostly `ReadTimeout`, reduce concurrency for a laptop run:

```bash
CONCURRENCY=8 DURATION_SECONDS=30 bash report/run_full_benchmark.sh
```

You can also increase the coordinator-to-reader timeout:

```bash
VECSCALE_COORDINATOR_READER_TIMEOUT_SECONDS=20 docker compose up -d --build
```

### Port `8000` already in use

Stop single-node mode or old Docker containers:

```bash
docker compose down
```

If a local Python server is running, stop it with `Ctrl+C`.
