"""QPS-vs-Nodes scaling evaluation for VecScaleDB.

Runs the QPS benchmark against the coordinator while varying the number of
active readers.  Produces a JSON results file that can be plotted to validate
the near-linear throughput scaling claim from the Milvus paper (Figure 10b).

Usage (from the project root, with the cluster already running):
    python -m vecscaledb.bench.scaling_eval \\
        --dataset data/sift1m/sift1m.hdf5 \\
        --coordinator-url http://127.0.0.1:8000 \\
        --reader-counts 1 2 3 4 5 \\
        --duration 30 \\
        --concurrency 32 \\
        --output scaling_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np

from vecscaledb.bench.benchmark_qps import run_benchmark, print_result
from vecscaledb.bench.load_dataset import load_sift1m


def _reader_service_names(count: int) -> list[str]:
    """Return docker-compose service names for `count` readers."""
    return [f"reader-{i}" for i in range(count)]


def _scale_readers(
    count: int,
    max_readers: int = 5,
    coordinator_url: str = "http://127.0.0.1:8000",
) -> None:
    """Scale reader services up/down using docker compose.

    Starts exactly `count` readers and stops the rest.
    """
    to_start = _reader_service_names(count)
    to_stop = [f"reader-{i}" for i in range(count, max_readers)]

    if to_stop:
        print(f"  Stopping readers: {', '.join(to_stop)}")
        subprocess.run(
            ["docker", "compose", "stop", *to_stop],
            capture_output=True,
        )

    if to_start:
        print(f"  Starting readers: {', '.join(to_start)}")
        subprocess.run(
            ["docker", "compose", "up", "-d", *to_start],
            capture_output=True,
        )

    # Wait for readers to become ready
    print(f"  Waiting for {count} reader(s) to be ready ...")
    for _ in range(60):
        time.sleep(2)
        ready_count = 0
        for service in to_start:
            result = subprocess.run(
                ["docker", "compose", "exec", service,
                 "python", "-c",
                 "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8200/ready', timeout=2).read()"],
                capture_output=True,
            )
            if result.returncode == 0:
                ready_count += 1
        if ready_count == count:
            print(f"  All {count} reader(s) ready.")
            _wait_for_coordinator_reader_view(coordinator_url, set(to_start))
            return
    print(f"  WARNING: Only {ready_count}/{count} readers became ready.")


def _wait_for_coordinator_reader_view(
    coordinator_url: str,
    expected_reader_ids: set[str],
) -> None:
    """Wait until coordinator metadata no longer contains stale readers."""
    print("  Waiting for coordinator reader registry to match active readers ...")
    for _ in range(30):
        try:
            with urllib.request.urlopen(
                f"{coordinator_url.rstrip('/')}/health",
                timeout=2,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            readers = {
                node["node_id"]
                for node in payload.get("nodes", [])
                if node.get("role") == "reader"
            }
            if readers == expected_reader_ids:
                print("  Coordinator reader registry is fresh.")
                return
        except Exception:
            pass
        time.sleep(1)
    print("  WARNING: Coordinator still may contain stale reader registrations.")


async def scaling_eval(
    coordinator_url: str,
    queries: np.ndarray,
    reader_counts: list[int],
    top_k: int,
    nprobe: int,
    concurrency: int,
    duration_seconds: float,
    auto_scale: bool,
) -> list[dict]:
    """Run QPS benchmarks for each reader count."""
    results = []

    for count in reader_counts:
        print(f"\n{'='*60}")
        print(f"Benchmark with {count} reader(s)")
        print(f"{'='*60}")

        if auto_scale:
            _scale_readers(count, coordinator_url=coordinator_url)
            # Extra settle time after scaling
            time.sleep(3)

        result = await run_benchmark(
            coordinator_url=coordinator_url,
            queries=queries,
            top_k=top_k,
            nprobe=nprobe,
            concurrency=concurrency,
            duration_seconds=duration_seconds,
        )
        result["readers"] = count
        results.append(result)
        print_result(result)

    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary table of scaling results."""
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Readers':>8} {'QPS':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Errors':>8}")
    print("-" * 70)

    base_qps = results[0]["qps"] if results else 1.0
    for r in results:
        speedup = r["qps"] / base_qps if base_qps > 0 else 0.0
        print(
            f"{r['readers']:>8} {r['qps']:>10.1f} {r['p50_ms']:>10.2f} "
            f"{r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f} {r['errors']:>8}"
            f"  (×{speedup:.2f})"
        )

    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        scaling_factor = last["qps"] / first["qps"] if first["qps"] > 0 else 0.0
        reader_ratio = last["readers"] / first["readers"]
        linearity = scaling_factor / reader_ratio if reader_ratio > 0 else 0.0
        print(f"\nScaling: {first['readers']}→{last['readers']} readers = "
              f"{scaling_factor:.2f}× QPS (linearity={linearity:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VecScaleDB QPS-vs-Nodes scaling evaluation."
    )
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="data/sift1m/sift1m.hdf5")
    parser.add_argument("--query-limit", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--reader-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Number of readers to benchmark (e.g. 1 2 3 4 5)",
    )
    parser.add_argument("--output", default="scaling_results.json")
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Automatically scale reader containers via docker compose",
    )
    args = parser.parse_args()

    _, queries, _ = load_sift1m(args.dataset)
    queries = queries[: args.query_limit]

    results = asyncio.run(
        scaling_eval(
            coordinator_url=args.coordinator_url,
            queries=queries,
            reader_counts=sorted(args.reader_counts),
            top_k=args.top_k,
            nprobe=args.nprobe,
            concurrency=args.concurrency,
            duration_seconds=args.duration,
            auto_scale=args.auto_scale,
        )
    )

    print_summary(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
