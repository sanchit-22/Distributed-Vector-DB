# VecScaleDB Runbook

This README contains the commands needed to run, test, and verify the project.

Project folder:

```powershell
cd "C:\Users\priya\OneDrive\Desktop\SEM 4\Distri\distriProject\Distributed-Vector-DB"
```

## 1. Create And Use Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If the virtual environment already exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Run All Automated Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected current result:

```text
66 passed
```

## 3. Run Single-Node API

This is the simpler Phase 1/2 mode.

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.node
```

Single-node URL:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 4. Run Full Distributed Cluster

Docker Desktop must be running.

Important: stop the single-node API first if it is running. Both single-node mode
and distributed coordinator-0 use port 8000.

```powershell
# In the terminal running: python -m vecscaledb.node
# press Ctrl+C
```

If `http://127.0.0.1:8000/docs` says `VecScaleDB - Single Node`, you are still
seeing the single-node server. After the cluster starts correctly, the same URL
should say `VecScaleDB - Coordinator`.

```powershell
docker compose up -d --build
```

Distributed API docs:

```text
http://127.0.0.1:8000/docs
```

For a quick manual demo where two inserted vectors become searchable
immediately, start the cluster with a small flush threshold:

```powershell
$env:VECSCALE_SEGMENT_FLUSH_THRESHOLD = "2"
docker compose up -d --build
```

For larger benchmark runs, use the normal command without that environment
variable, or reset it:

```powershell
Remove-Item Env:\VECSCALE_SEGMENT_FLUSH_THRESHOLD -ErrorAction SilentlyContinue
docker compose up -d --build
```

For live logs, like watching the server terminal:
docker compose logs -f

Services:

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

Check containers:

```powershell
docker compose ps
```

Expected: all coordinators, writer, etcd, and all five readers are `Up`; readers should show `(healthy)`.

Stop cluster:

```powershell
docker compose down
```

## 5. Health And Readiness Checks

Coordinators:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
```

Writer:

```powershell
Invoke-RestMethod http://127.0.0.1:8100/health
```

Readers:

```powershell
Invoke-RestMethod http://127.0.0.1:8200/ready
Invoke-RestMethod http://127.0.0.1:8201/ready
Invoke-RestMethod http://127.0.0.1:8202/ready
Invoke-RestMethod http://127.0.0.1:8203/ready
Invoke-RestMethod http://127.0.0.1:8204/ready
```

Segments:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/segments
```

## 6. Insert Test Vectors

Small insert into the writer. Use the writer URL on port 8100 for inserts:

```powershell
$dim = 128
$v1 = @(1.0) + @(0.0) * ($dim - 1)
$v2 = @(0.0, 1.0) + @(0.0) * ($dim - 2)

$body = @{
  ids = @(1, 2)
  vectors = @($v1, $v2)
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8100/insert -Body $body -ContentType "application/json"
```

Expected for the quick demo threshold:

```text
inserted        : 2
segment_flushed : True
```

Important: vectors become visible to distributed readers after they are flushed into a segment. Default flush threshold is 50,000 vectors. For a quick manual demo, start Docker with `VECSCALE_SEGMENT_FLUSH_THRESHOLD=2` as shown above.

## 7. Search Through Coordinator

Use the coordinator URL on port 8000 for distributed search:

```powershell
$dim = 128
$query = @(1.0) + @(0.0) * ($dim - 1)

$body = @{
  query = $query
  top_k = 5
  nprobe = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -Body $body -ContentType "application/json"
```

Expected:

```text
ids contains the inserted ID
distances contains 0.0 for the exact match
incomplete is False
```

Search with a trace ID:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -Headers @{ "x-trace-id" = "manual-test-1" } `
  -Body $body `
  -ContentType "application/json"
```

## 8. Metrics And Logs

Metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
Invoke-RestMethod http://127.0.0.1:8100/metrics
Invoke-RestMethod http://127.0.0.1:8200/metrics
```

Logs:

```powershell
docker compose logs --tail=50 coordinator-0
docker compose logs --tail=50 writer-0
docker compose logs --tail=50 reader-0
```

Search for one trace ID in logs:

```powershell
docker compose logs coordinator-0 | Select-String -Pattern "manual-test-1"
```

## 9. Fault Tolerance Checks

Reader restart:

```powershell
docker compose stop reader-3
docker compose start reader-3
Start-Sleep -Seconds 12
Invoke-RestMethod http://127.0.0.1:8203/ready
```

Writer crash/restart:

```powershell
docker compose kill writer-0
docker compose start writer-0
Start-Sleep -Seconds 8
Invoke-RestMethod http://127.0.0.1:8100/health
```

Coordinator failover:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health

docker compose kill coordinator-0
Start-Sleep -Seconds 15

Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health

docker compose start coordinator-0
```

## 10. SIFT Dataset And Benchmark

Download SIFT1M:

```powershell
bash data/download_sift.sh
```

Load SIFT1M into writer:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.load_dataset --dataset data\sift1m\sift1m.hdf5 --writer-url http://127.0.0.1:8100 --batch-size 10000
```

Run QPS benchmark:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.benchmark_qps --dataset data\sift1m\sift1m.hdf5 --coordinator-url http://127.0.0.1:8000 --concurrency 32 --duration 60
```

Run recall benchmark:

```powershell
.\.venv\Scripts\python.exe -m vecscaledb.bench.recall_eval --host 127.0.0.1 --port 8000 --dataset data\sift1m\sift1m.hdf5
```

## 11. Guide Files

Detailed guides are inside the project folder:

```text
Distributed-Vector-DB\PHASE0_PHASE1_GUIDE.md
Distributed-Vector-DB\PHASE2_PHASE3_GUIDE.md
Distributed-Vector-DB\PHASE4_PHASE5_GUIDE.md
Distributed-Vector-DB\PHASE6_PHASE7_GUIDE.md
Distributed-Vector-DB\PHASE8_GUIDE.md
Distributed-Vector-DB\PROJECT_NOTES_ROADMAP.md
Distributed-Vector-DB\benchmark_report.md
```
