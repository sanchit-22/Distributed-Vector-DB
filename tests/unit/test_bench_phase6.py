"""Tests for Phase 6 benchmark utilities."""

import asyncio

import h5py
import httpx
import numpy as np
import pytest

from vecscaledb.bench import benchmark_qps
from vecscaledb.bench.load_dataset import iter_hdf5_train_batches


def test_iter_hdf5_train_batches_streams_requested_slice(tmp_path):
    dataset = tmp_path / "tiny.hdf5"
    values = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    with h5py.File(dataset, "w") as f:
        f.create_dataset("train", data=values)

    batches = list(
        iter_hdf5_train_batches(str(dataset), batch_size=3, offset=2, limit=5)
    )

    assert [start for start, _ in batches] == [2, 5]
    np.testing.assert_array_equal(batches[0][1], values[2:5])
    np.testing.assert_array_equal(batches[1][1], values[5:7])


def test_percentile_interpolates_values():
    assert benchmark_qps.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert benchmark_qps.percentile([10.0, 20.0], 95) == pytest.approx(19.5)
    assert benchmark_qps.percentile([], 99) == 0.0


def test_run_benchmark_reports_qps(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, path, json):
            return httpx.Response(
                200,
                json={"ids": [1], "distances": [0.0], "incomplete": False},
                request=httpx.Request("POST", "http://test/search"),
            )

    monkeypatch.setattr(benchmark_qps.httpx, "AsyncClient", FakeAsyncClient)
    queries = np.eye(4, dtype=np.float32)

    result = asyncio.run(
        benchmark_qps.run_benchmark(
            coordinator_url="http://test",
            queries=queries,
            top_k=1,
            nprobe=1,
            concurrency=2,
            duration_seconds=0.01,
        )
    )

    assert result["total_queries"] > 0
    assert result["attempts"] >= result["total_queries"]
    assert result["errors"] == 0
    assert result["error_rate"] == 0.0
    assert result["qps"] > 0
    assert result["p99_ms"] >= result["p50_ms"]
