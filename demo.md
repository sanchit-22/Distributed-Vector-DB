# VecScaleDB Evaluation Demo Script

This file is a step-by-step demo guide for evaluation. It shows the project in
two modes:

- single-node mode: one Python server, easiest to understand,
- distributed mode: Docker cluster with coordinators, writer, readers, and etcd.

Use PowerShell on Windows. Run commands from the project folder unless a step
says otherwise.

## 0. Demo Goal

By the end of the demo, you should be able to show:

- the API starts,
- vectors can be inserted,
- nearest-neighbor search works,
- data survives restart in single-node mode,
- the distributed cluster starts,
- writer, readers, and coordinators work together,
- health, segments, logs, and metrics are visible,
- the cluster keeps searching when one reader or coordinator is stopped.

## 1. Go To Project Folder

```powershell
cd "C:\Users\priya\OneDrive\Desktop\SEM 4\Distri\distriProject\Distributed-Vector-DB"
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Run Automated Tests

This proves the implemented functions pass the local test suite.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected result:

```text
66 passed
```

Small note: two dependency warnings can appear. They are not test failures.

## 3. Demo Single-Node Mode

Single-node mode is the simple mode. One Python process does everything:

- accepts inserts,
- writes WAL,
- stores vectors,
- searches vectors,
- exposes API docs.

### 3.1 Start Single-Node API

Open a terminal and run:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.node
```

Keep this terminal open.

Open the browser:

```text
http://127.0.0.1:8000/docs
```

Expected page title:

```text
VecScaleDB - Single Node
```

### 3.2 Check Health

Open a second PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected fields:

```text
status           : ok
ntotal           : number of stored vectors
segments         : number of persisted segment files
memtable_size    : vectors currently in memory
wal_lsn          : latest WAL write number
active_snapshots : current active read snapshots
```

Meaning:

- `ntotal` tells how many vectors the database knows.
- `wal_lsn` proves write-ahead logging is active.
- `segments` proves flushed storage files are loaded.

### 3.3 Insert Two Vectors

```powershell
$dim = 128
$v1 = @(1.0) + (@(0.0) * 127)
$v2 = @(0.0, 1.0) + (@(0.0) * 126)

$body = @{
  ids = @(101, 102)
  vectors = @($v1, $v2)
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/insert `
  -Body $body `
  -ContentType "application/json"
```

Expected:

```text
inserted : 2
ntotal   : increases by 2
lsn      : increases
```

Meaning:

- vector `101` is `[1.0, 0.0, 0.0, ...]`,
- vector `102` is `[0.0, 1.0, 0.0, ...]`,
- the API accepted both vectors.

### 3.4 Search Single Node

Search for the first vector:

```powershell
$query = @(1.0) + (@(0.0) * 127)

$body = @{
  query = $query
  top_k = 2
  nprobe = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Body $body `
  -ContentType "application/json"
```

Expected:

```text
ids        distances
---        ---------
{101, ...} {0.0, ...}
```

Meaning:

- ID `101` appears first because the query exactly matches it.
- Distance `0.0` means exact match.

### 3.5 Test Single-Node Durability

Stop the single-node server with:

```text
Ctrl+C
```

Start it again:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.node
```

In the second terminal, run the same search again:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Body $body `
  -ContentType "application/json"
```

Expected: ID `101` still appears.

Meaning:

- The data survived restart.
- WAL recovery and segment loading are working.

### 3.6 Stop Single-Node Before Distributed Mode

Distributed mode also uses port `8000`, so stop single-node first:

```text
Ctrl+C
```

## 4. Demo Distributed Cluster

Distributed mode runs multiple services:

```text
coordinator-0  http://127.0.0.1:8000
coordinator-1  http://127.0.0.1:8001
coordinator-2  http://127.0.0.1:8002
writer-0       http://127.0.0.1:8100
reader-0       http://127.0.0.1:8200
reader-1       http://127.0.0.1:8201
reader-2       http://127.0.0.1:8202
reader-3       http://127.0.0.1:8203
reader-4       http://127.0.0.1:8204
etcd           http://127.0.0.1:2379
```

Important idea:

- insert goes to the writer on port `8100`,
- search goes to the coordinator on port `8000`,
- readers load segments and answer searches,
- etcd stores cluster metadata.

### 4.1 Start Distributed Cluster

Docker Desktop must be running.

For an evaluation demo, use a small flush threshold so two inserted vectors
become searchable immediately:

```powershell
$env:VECSCALE_SEGMENT_FLUSH_THRESHOLD = "2"
docker compose up -d --build
```

This is the distributed equivalent of starting the server.

### 4.2 Check Containers

```powershell
docker compose ps
```

Expected:

- all services are `Up`,
- all readers show `(healthy)`.

### 4.3 Open Distributed API Docs

Open:

```text
http://127.0.0.1:8000/docs
```

Expected page title:

```text
VecScaleDB - Coordinator
```

If it says `VecScaleDB - Single Node`, then single-node is still running and
must be stopped.

## 5. Distributed Health Checks

### 5.1 Coordinators

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
```

Expected:

- cluster node list appears,
- `ring_size` is greater than 0,
- one coordinator should be leader.

### 5.2 Writer

```powershell
Invoke-RestMethod http://127.0.0.1:8100/health
```

Expected:

```text
status : ok
wal_lsn : latest write number
segments : persisted segment count
```

### 5.3 Readers

```powershell
Invoke-RestMethod http://127.0.0.1:8200/ready
Invoke-RestMethod http://127.0.0.1:8201/ready
Invoke-RestMethod http://127.0.0.1:8202/ready
Invoke-RestMethod http://127.0.0.1:8203/ready
Invoke-RestMethod http://127.0.0.1:8204/ready
```

Expected:

```text
status : ready
segments_loaded : number of loaded segments
ntotal : number of loaded vectors
```

## 6. Distributed Insert

Insert two new vectors into the writer:

```powershell
$dim = 128
$v1 = @(0.25, 0.5, 0.75, 1.0) + (@(0.0) * 124)
$v2 = @(1.0, 0.75, 0.5, 0.25) + (@(0.0) * 124)

$body = @{
  ids = @(990001, 990002)
  vectors = @($v1, $v2)
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8100/insert `
  -Body $body `
  -ContentType "application/json"
```

Expected:

```text
inserted        : 2
segment_flushed : True
lsn             : increases
```

Meaning:

- the writer accepted the vectors,
- WAL wrote the operation,
- because flush threshold is `2`, a searchable segment was created.

Wait for readers to refresh:

```powershell
Start-Sleep -Seconds 6
```

## 7. Distributed Search

Search through the coordinator:

```powershell
$body = @{
  query = $v1
  top_k = 3
  nprobe = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Body $body `
  -ContentType "application/json"
```

Expected:

```text
ids contains 990001
distances contains 0.0
incomplete is False
```

Meaning:

- coordinator received the search,
- readers searched loaded segments,
- coordinator merged results,
- `incomplete: False` means no reader failure affected this query.

## 8. Show Segments

```powershell
Invoke-RestMethod http://127.0.0.1:8000/segments
```

Expected:

- a list of segment metadata,
- each segment has `segment_id`, `num_vectors`, `snapshot_id`, and `path`.

Meaning:

- segments are the persisted search files,
- readers load these files from shared storage.

## 9. Show Metrics

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
Invoke-RestMethod http://127.0.0.1:8100/metrics
Invoke-RestMethod http://127.0.0.1:8200/metrics
```

Expected metric names include:

```text
vecscaledb_request_total
vecscaledb_request_duration_seconds_count
vecscaledb_search_duration_seconds_count
vecscaledb_registered_readers
vecscaledb_segments_loaded
```

Meaning:

- metrics show request counts, timings, loaded segments, readers, and cluster state.

## 10. Show Logs

Recent logs:

```powershell
docker compose logs --tail=50 coordinator-0
docker compose logs --tail=50 writer-0
docker compose logs --tail=50 reader-0
```

Live logs:

```powershell
docker compose logs -f
```

Stop watching live logs:

```text
Ctrl+C
```

This only stops log viewing. It does not stop Docker containers.

## 11. Trace ID Demo

This proves request tracing works.

```powershell
$body = @{
  query = $v1
  top_k = 3
  nprobe = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Headers @{ "x-trace-id" = "eval-demo-trace-1" } `
  -Body $body `
  -ContentType "application/json"
```

Now search logs for that trace:

```powershell
docker compose logs coordinator-0 | Select-String -Pattern "eval-demo-trace-1"
```

Expected:

- log lines containing `eval-demo-trace-1`.

Meaning:

- trace IDs can connect one user request to logs.

## 12. Invalid Input Demo

This proves the API returns clean client errors.

### 12.1 Bad Writer Insert

```powershell
$badBody = @{
  ids = @(1, 2)
  vectors = @(@(1.0, 2.0), @(3.0))
} | ConvertTo-Json -Depth 5

try {
  Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8100/insert `
    -Body $badBody `
    -ContentType "application/json"
} catch {
  [int]$_.Exception.Response.StatusCode
}
```

Expected:

```text
400
```

### 12.2 Bad Search Limit

```powershell
$badSearch = @{
  query = $v1
  top_k = 0
  nprobe = 32
} | ConvertTo-Json -Depth 5

try {
  Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/search `
    -Body $badSearch `
    -ContentType "application/json"
} catch {
  [int]$_.Exception.Response.StatusCode
}
```

Expected:

```text
422
```

Meaning:

- `400` means the request shape reached app validation and was rejected,
- `422` means schema validation rejected the request before app logic ran.

## 13. Reader Failure Demo

Stop one reader:

```powershell
docker compose stop reader-3
```

Search again through the coordinator:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Body $body `
  -ContentType "application/json"
```

Expected:

- the search should still return results,
- no server crash should happen.

Restart the reader:

```powershell
docker compose start reader-3
Start-Sleep -Seconds 8
Invoke-RestMethod http://127.0.0.1:8203/ready
```

Expected:

```text
status : ready
```

Meaning:

- the system can continue searching while one reader is unavailable,
- reader can rejoin and reload segments.

## 14. Coordinator Failover Demo

Stop coordinator-0:

```powershell
docker compose stop coordinator-0
```

Search through coordinator-1:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8001/search `
  -Body $body `
  -ContentType "application/json"
```

Expected:

- search still returns results.

Restart coordinator-0:

```powershell
docker compose start coordinator-0
Start-Sleep -Seconds 5
Invoke-RestMethod http://127.0.0.1:8000/health
```

Meaning:

- another coordinator can continue serving requests,
- stopped coordinator can rejoin.

## 15. Optional Benchmark Demo

This requires the SIFT1M dataset file. If the dataset is missing, skip this in
evaluation and mention it is an external large dataset.

Download:

```powershell
bash data/download_sift.sh
```

Load dataset:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.load_dataset `
  --dataset data\sift1m\sift1m.hdf5 `
  --writer-url http://127.0.0.1:8100 `
  --batch-size 10000
```

Run QPS benchmark:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.benchmark_qps `
  --dataset data\sift1m\sift1m.hdf5 `
  --coordinator-url http://127.0.0.1:8000 `
  --concurrency 32 `
  --duration 60
```

Run recall benchmark:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.recall_eval `
  --host 127.0.0.1 `
  --port 8000 `
  --dataset data\sift1m\sift1m.hdf5
```

## 16. Stop Everything After Demo

Stop distributed cluster:

```powershell
docker compose down
```

Remove the quick-demo environment variable:

```powershell
Remove-Item Env:\VECSCALE_SEGMENT_FLUSH_THRESHOLD -ErrorAction SilentlyContinue
```

## 17. Short Explanation For Evaluation

You can say:

```text
This project is a distributed vector database prototype. It stores numerical
vectors, searches for nearest vectors, writes data safely using a WAL, persists
flushed vectors as segment files, and uses a Docker cluster with coordinators,
a writer, readers, and etcd metadata. The coordinator routes search, the writer
handles inserts, and readers load segments to answer queries. The project also
includes health checks, readiness checks, metrics, structured logs, trace IDs,
and fault-tolerance demos.
```

## 18. Common Mistakes

### Browser shows `VecScaleDB - Single Node`

You are still running single-node mode. Stop it with `Ctrl+C`, then start Docker.

### Browser shows `Method Not Allowed` for `/search`

`/search` is a POST endpoint. Use Swagger `/docs` or PowerShell.

### Insert works but distributed search does not find new vectors

Readers can only search flushed segments. For manual demo, start Docker with:

```powershell
$env:VECSCALE_SEGMENT_FLUSH_THRESHOLD = "2"
docker compose up -d --build
```

### Docker says port 8000 is already in use

Stop single-node mode first:

```text
Ctrl+C
```

### Live logs keep running

Press:

```text
Ctrl+C
```

This stops log viewing only, not the cluster.
