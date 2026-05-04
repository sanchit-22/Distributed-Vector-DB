# VecScaleDB ANN Benchmark Report Tools

This folder contains the scripts used to generate the project benchmark report
PDF without adding generated benchmark code to the main source tree.

## Files

```text
report/collect_metrics.py
report/generate_report.py
report/run_dataset_metrics.py
report/run_all_dataset_metrics.sh
report/run_report.sh
report/run_sift1m_scaling.sh
report/download_deep1m.sh
report/vecscaledb_ann_benchmark_report.pdf
report/latex/vecscaledb_report.tex
report/latex/vecscaledb_report.pdf
```

`collect_metrics.py` inspects local datasets and existing benchmark result JSON
files. It writes a normalized metrics file to:

```text
report/build/metrics.json
```

`generate_report.py` reads that metrics file and creates:

```text
report/vecscaledb_ann_benchmark_report.pdf
```

The PDF generator is intentionally self-contained. It uses only the Python
standard library, so no extra PDF package is required.

The `report/latex/` folder contains a separate LaTeX report with architecture
diagrams, request-flow diagrams, dataset tables, benchmark tables, and pgfplots
charts. It is the recommended editable report format for submission/viva.

## Generate The Report

From the project root:

```bash
source venv/bin/activate
bash report/run_report.sh
```

The output PDF will be:

```text
report/vecscaledb_ann_benchmark_report.pdf
```

## Generate The LaTeX Report

From the project root:

```bash
cd report/latex
pdflatex vecscaledb_report.tex
pdflatex vecscaledb_report.tex
```

The output PDF will be:

```text
report/latex/vecscaledb_report.pdf
```

## Expected Dataset Paths

```text
data/sift1m/sift1m.hdf5
data/deep1m/deep1m.hdf5
```

SIFT10K is treated as the first 10,000 train vectors from SIFT1M, matching the
project's fast correctness evaluation path.

Deep1M is treated as the first 1,000,000 train vectors from the public
ANN-Benchmarks 96D Deep image HDF5 file. That public file is named
`deep-image-96-angular.hdf5`; VecScaleDB evaluates those vectors with L2 because
the current engine is IVF_FLAT/L2. For recall, the script skips Deep recall by
default unless you explicitly force it, because the public ground truth is
angular rather than L2.

Download Deep:

```bash
bash report/download_deep1m.sh
```

## Benchmark Inputs

The collector reads these optional result files when present:

```text
scaling_results.json
report/results/sift10k_scaling_results.json
report/results/sift1m_scaling_results.json
report/results/deep1m_scaling_results.json
report/results/sift10k_metrics.json
report/results/sift1m_metrics.json
report/results/deep1m_metrics.json
report/results/recall_results.json
```

The current codebase already has `scaling_results.json`, which is used as the
SIFT1M scaling run unless a dataset-specific result file is present.

## Running Benchmarks

### One Dataset At A Time

SIFT10K, for quick correctness and smoke metrics:

```bash
docker compose down
VECSCALE_SEGMENT_FLUSH_THRESHOLD=10000 docker compose up -d --build

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

SIFT1M scaling:

```bash
bash report/run_sift1m_scaling.sh
```

You can shorten test runs without editing the script:

```bash
DURATION_SECONDS=10 CONCURRENCY=16 QUERY_LIMIT=200 bash report/run_sift1m_scaling.sh
```

Deep1M requires a 96D cluster and clean runtime state:

```bash
bash report/download_deep1m.sh
docker compose down
rm -rf shared_storage wal
mkdir -p shared_storage wal
VECSCALE_DIM=96 docker compose up -d --build

python report/run_dataset_metrics.py \
  --dataset deep1m \
  --load \
  --qps \
  --scaling \
  --auto-scale \
  --duration 60 \
  --concurrency 32 \
  --query-limit 1000 \
  --output report/results/deep1m_metrics.json
```

Important: keep `VECSCALE_DIM=96` in the environment whenever reader containers
are restarted manually. The metrics runner sets this automatically before
`--auto-scale`, but direct `docker compose up -d reader-0` commands need the same
environment variable for Deep1M.

After any benchmark run:

```bash
bash report/run_report.sh
```

### All Proposal Datasets

Run SIFT10K and SIFT1M:

```bash
bash report/run_all_dataset_metrics.sh
```

For cleaner numbers that reset runtime state before each dataset:

```bash
RESET_STATE=1 bash report/run_all_dataset_metrics.sh
```

Include Deep1M too:

```bash
bash report/download_deep1m.sh
RUN_DEEP=1 RESET_STATE=1 bash report/run_all_dataset_metrics.sh
```

The `RESET_STATE=1` flag is required before switching from 128D SIFT data to
96D Deep data because old 128D segment files cannot be loaded by a 96D cluster.

SIFT10K correctness is normally run in single-node mode:

```bash
python -m vecscaledb.bench.recall_eval \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset data/sift1m/sift1m.hdf5
```

Deep1M support is represented in the report tooling, but the dataset and loader
must be supplied locally at `data/deep1m/deep1m.hdf5`.
