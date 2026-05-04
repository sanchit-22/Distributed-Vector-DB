"""Sustained QPS benchmark for coordinator search."""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx
import numpy as np

from vecscaledb.bench.load_dataset import load_sift1m


def percentile(values: list[float], pct: float) -> float:
    """Return a percentile from a non-empty sorted-or-unsorted list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


async def run_benchmark(
    coordinator_url: str,
    queries: np.ndarray,
    top_k: int,
    nprobe: int,
    concurrency: int,
    duration_seconds: float,
) -> dict:
    """Run a fixed-duration concurrent search benchmark.

    Args:
        coordinator_url: Base URL of the coordinator, for example
            ``http://127.0.0.1:8000``.
        queries: Query vectors with shape ``[N, dim]``.
        top_k: Search result count.
        nprobe: IVF search probe count.
        concurrency: Number of worker coroutines issuing requests.
        duration_seconds: Benchmark duration.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if queries.ndim != 2 or queries.shape[0] == 0:
        raise ValueError("queries must be a non-empty [N, dim] array.")

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    deadline = time.perf_counter() + duration_seconds
    latencies_ms: list[float] = []
    errors = 0
    attempts = 0
    incomplete_errors = 0
    request_errors = 0

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal attempts, errors, incomplete_errors, request_errors
        while time.perf_counter() < deadline:
            query = queries[random.randrange(queries.shape[0])]
            body = {
                "query": query.tolist(),
                "top_k": top_k,
                "nprobe": nprobe,
            }
            started = time.perf_counter()
            attempts += 1
            try:
                response = await client.post("/search", json=body)
                response.raise_for_status()
                payload = response.json()
                if payload.get("incomplete", False):
                    incomplete_errors += 1
                    errors += 1
                else:
                    latencies_ms.append((time.perf_counter() - started) * 1000.0)
            except Exception:
                request_errors += 1
                errors += 1

    timeout = httpx.Timeout(10.0, connect=5.0)
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=coordinator_url.rstrip("/"),
        timeout=timeout,
    ) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
    elapsed = time.perf_counter() - started

    total_queries = len(latencies_ms)
    return {
        "total_queries": total_queries,
        "attempts": attempts,
        "errors": errors,
        "incomplete_errors": incomplete_errors,
        "request_errors": request_errors,
        "error_rate": errors / attempts if attempts > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "qps": total_queries / elapsed if elapsed > 0 else 0.0,
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
    }


def print_result(result: dict) -> None:
    print(
        "total_queries={total_queries} attempts={attempts} errors={errors} "
        "error_rate={error_rate:.2%} "
        "elapsed={elapsed_seconds:.2f}s qps={qps:.2f} "
        "p50={p50_ms:.2f}ms p95={p95_ms:.2f}ms p99={p99_ms:.2f}ms".format(
            **result
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VecScaleDB QPS benchmark.")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="data/sift1m/sift1m.hdf5")
    parser.add_argument("--query-limit", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    _, queries, _ = load_sift1m(args.dataset)
    queries = queries[: args.query_limit]
    result = asyncio.run(
        run_benchmark(
            coordinator_url=args.coordinator_url,
            queries=queries,
            top_k=args.top_k,
            nprobe=args.nprobe,
            concurrency=args.concurrency,
            duration_seconds=args.duration,
        )
    )
    print_result(result)


if __name__ == "__main__":
    main()
