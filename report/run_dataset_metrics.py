"""Run dataset-specific VecScaleDB metrics and write JSON results.

This script supports the datasets from the project proposal:

- SIFT10K: first 10,000 train vectors from SIFT1M, 128D, L2
- SIFT1M: 1,000,000 train vectors from SIFT1M, 128D, L2
- Deep1M: first 1,000,000 train vectors from Deep 96D HDF5, evaluated with L2

It can load vectors, run QPS, run reader scaling, and compute Recall@k when
ground truth is compatible with the requested metric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import httpx
import numpy as np

from vecscaledb.bench.benchmark_qps import run_benchmark
from vecscaledb.bench.scaling_eval import scaling_eval


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "report" / "results"


DATASETS = {
    "sift10k": {
        "display_name": "SIFT10K",
        "path": ROOT / "data" / "sift1m" / "sift1m.hdf5",
        "dim": 128,
        "metric": "L2",
        "ground_truth_metric": "L2",
        "train_limit": 10_000,
        "query_limit": 100,
        "default_output": RESULTS_DIR / "sift10k_metrics.json",
        "notes": ["Derived from SIFT1M train[:10000]."],
    },
    "sift1m": {
        "display_name": "SIFT1M",
        "path": ROOT / "data" / "sift1m" / "sift1m.hdf5",
        "dim": 128,
        "metric": "L2",
        "ground_truth_metric": "L2",
        "train_limit": 1_000_000,
        "query_limit": 1000,
        "default_output": RESULTS_DIR / "sift1m_metrics.json",
        "notes": ["Primary performance benchmark."],
    },
    "deep1m": {
        "display_name": "Deep1M",
        "path": ROOT / "data" / "deep1m" / "deep1m.hdf5",
        "dim": 96,
        "metric": "L2",
        "ground_truth_metric": "Angular",
        "train_limit": 1_000_000,
        "query_limit": 1000,
        "default_output": RESULTS_DIR / "deep1m_metrics.json",
        "notes": [
            "Uses the first 1,000,000 train vectors from deep-image-96-angular.hdf5.",
            "ANN-Benchmarks Deep ground truth is angular; L2 recall is skipped unless forced.",
        ],
    },
}


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    output = Path(args.output) if args.output else spec["default_output"]
    output.parent.mkdir(parents=True, exist_ok=True)

    train_limit = args.train_limit or spec["train_limit"]
    query_limit = args.query_limit or spec["query_limit"]
    path = Path(args.path) if args.path else spec["path"]
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")

    metrics: dict[str, Any] = {
        "dataset": spec["display_name"],
        "dataset_key": args.dataset,
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "dim": spec["dim"],
        "metric": spec["metric"],
        "ground_truth_metric": args.ground_truth_metric or spec["ground_truth_metric"],
        "train_limit": train_limit,
        "query_limit": query_limit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": list(spec["notes"]),
        "hdf5": inspect_hdf5(path),
    }

    # scaling_eval calls `docker compose up -d reader-*` when --auto-scale is
    # enabled. docker-compose.yml interpolates VECSCALE_DIM at container start,
    # so keep that environment variable aligned with the dataset dimension.
    os.environ["VECSCALE_DIM"] = str(spec["dim"])

    if args.load:
        metrics["load"] = load_into_writer(
            path=path,
            writer_url=args.writer_url,
            batch_size=args.batch_size,
            limit=train_limit,
            timeout_seconds=args.load_timeout,
        )

    queries, neighbors = load_queries(path, query_limit)

    if args.qps:
        metrics["qps"] = asyncio.run(
            run_benchmark(
                coordinator_url=args.coordinator_url,
                queries=queries,
                top_k=args.top_k,
                nprobe=args.nprobe,
                concurrency=args.concurrency,
                duration_seconds=args.duration,
            )
        )

    if args.scaling:
        metrics["scaling"] = asyncio.run(
            scaling_eval(
                coordinator_url=args.coordinator_url,
                queries=queries,
                reader_counts=args.reader_counts,
                top_k=args.top_k,
                nprobe=args.nprobe,
                concurrency=args.concurrency,
                duration_seconds=args.duration,
                auto_scale=args.auto_scale,
            )
        )

    if args.recall:
        gt_metric = metrics["ground_truth_metric"].lower()
        requested_metric = spec["metric"].lower()
        if gt_metric != requested_metric and not args.force_recall:
            metrics["recall"] = {
                "skipped": True,
                "reason": (
                    f"Ground truth metric is {metrics['ground_truth_metric']} but "
                    f"VecScaleDB run metric is {spec['metric']}."
                ),
            }
        else:
            metrics["recall"] = compute_recall(
                coordinator_url=args.coordinator_url,
                queries=queries[: args.recall_queries],
                neighbors=neighbors[: args.recall_queries],
                top_ks=args.recall_k,
                nprobe=args.nprobe,
            )

    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VecScaleDB dataset metrics.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--path", default=None, help="Override HDF5 path.")
    parser.add_argument("--writer-url", default="http://127.0.0.1:8100")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--load-timeout", type=float, default=300.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--reader-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--recall-k", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--recall-queries", type=int, default=100)
    parser.add_argument("--ground-truth-metric", default=None)
    parser.add_argument("--force-recall", action="store_true")
    parser.add_argument("--load", action="store_true", help="Load vectors into writer.")
    parser.add_argument("--qps", action="store_true", help="Run single QPS benchmark.")
    parser.add_argument("--scaling", action="store_true", help="Run QPS-vs-readers benchmark.")
    parser.add_argument("--recall", action="store_true", help="Run Recall@k benchmark.")
    parser.add_argument("--auto-scale", action="store_true", help="Scale Docker readers during scaling.")
    return parser.parse_args()


def inspect_hdf5(path: Path) -> dict[str, Any]:
    keys: dict[str, Any] = {}
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            keys[key] = {"shape": list(h5[key].shape), "dtype": str(h5[key].dtype)}
    return {"keys": keys}


def iter_train_batches(
    path: Path,
    batch_size: int,
    limit: int,
) -> Iterator[tuple[int, np.ndarray]]:
    with h5py.File(path, "r") as h5:
        train = h5["train"]
        stop = min(int(train.shape[0]), limit)
        for start in range(0, stop, batch_size):
            end = min(start + batch_size, stop)
            yield start, np.asarray(train[start:end], dtype=np.float32)


def load_into_writer(
    path: Path,
    writer_url: str,
    batch_size: int,
    limit: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    inserted = 0
    started = time.perf_counter()
    last_response: dict[str, Any] = {}

    with httpx.Client(base_url=writer_url.rstrip("/"), timeout=timeout_seconds) as client:
        for source_start, vectors in iter_train_batches(path, batch_size, limit):
            ids = np.arange(source_start, source_start + len(vectors), dtype=np.int64)
            response = client.post(
                "/insert",
                json={"ids": ids.tolist(), "vectors": vectors.tolist()},
            )
            response.raise_for_status()
            last_response = response.json()
            inserted += len(vectors)
            elapsed = time.perf_counter() - started
            rate = inserted / elapsed if elapsed > 0 else 0.0
            print(
                f"loaded={inserted} lsn={last_response.get('lsn')} "
                f"segments={last_response.get('segments')} rate={rate:.1f}/s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return {
        "inserted": inserted,
        "elapsed_seconds": elapsed,
        "vectors_per_second": inserted / elapsed if elapsed > 0 else 0.0,
        "last_response": last_response,
    }


def load_queries(path: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        queries = np.asarray(h5["test"][:limit], dtype=np.float32)
        neighbors = np.asarray(h5["neighbors"][:limit], dtype=np.int64)
    return queries, neighbors


def compute_recall(
    coordinator_url: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_ks: list[int],
    nprobe: int,
) -> dict[str, Any]:
    max_k = max(top_ks)
    predictions = []
    latencies_ms = []
    errors = 0
    with httpx.Client(base_url=coordinator_url.rstrip("/"), timeout=60.0) as client:
        for query in queries:
            started = time.perf_counter()
            try:
                response = client.post(
                    "/search",
                    json={
                        "query": query.tolist(),
                        "top_k": max_k,
                        "nprobe": nprobe,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                pred = payload.get("ids", [])
                pred += [-1] * (max_k - len(pred))
                predictions.append(pred[:max_k])
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
            except Exception:
                errors += 1
                predictions.append([-1] * max_k)

    pred_arr = np.asarray(predictions, dtype=np.int64)
    recall_values = {}
    for k in top_ks:
        recalls = []
        for i in range(len(pred_arr)):
            gt_set = set(neighbors[i, :k].tolist())
            pred_set = set(pred_arr[i, :k].tolist())
            recalls.append(len(gt_set & pred_set) / k)
        recall_values[f"recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0

    return {
        "skipped": False,
        "queries": int(len(queries)),
        "errors": errors,
        "nprobe": nprobe,
        "latency_ms_avg": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        **recall_values,
    }


if __name__ == "__main__":
    main()
