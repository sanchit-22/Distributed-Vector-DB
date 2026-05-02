"""Dataset loading and bulk ingest utilities for ANN benchmark HDF5 files."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

import h5py
import httpx
import numpy as np


def load_sift1m(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the SIFT1M dataset from an HDF5 file.

    Returns:
        (train_vectors, query_vectors, ground_truth_neighbors)
        - train: [1_000_000, 128] float32
        - test: [10_000, 128] float32
        - neighbors: [10_000, 100] int32
    """
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    return train, test, neighbors


def load_sift10k(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 10K-vector slice of SIFT1M for fast development testing."""
    train, test, neighbors = load_sift1m(path)
    return train[:10_000], test[:100], neighbors[:100]


def iter_hdf5_train_batches(
    path: str,
    batch_size: int,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield contiguous train-vector batches without loading the full file."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")

    with h5py.File(path, "r") as f:
        train = f["train"]
        total = int(train.shape[0])
        stop = total if limit is None else min(total, offset + limit)
        for start in range(offset, stop, batch_size):
            end = min(start + batch_size, stop)
            yield start, np.asarray(train[start:end], dtype=np.float32)


def load_into_writer(
    dataset_path: str,
    writer_url: str,
    *,
    batch_size: int = 10_000,
    limit: int | None = None,
    start_id: int = 0,
    offset: int = 0,
    timeout_seconds: float = 300.0,
) -> dict:
    """Insert HDF5 train vectors into a Writer node in batches."""
    inserted = 0
    started = time.perf_counter()
    last_response: dict = {}

    with httpx.Client(base_url=writer_url, timeout=timeout_seconds) as client:
        for source_start, vectors in iter_hdf5_train_batches(
            dataset_path,
            batch_size,
            limit=limit,
            offset=offset,
        ):
            ids = np.arange(
                start_id + source_start,
                start_id + source_start + len(vectors),
                dtype=np.int64,
            )
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
                f"inserted={inserted} last_lsn={last_response.get('lsn')} "
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-load SIFT vectors into VecScaleDB.")
    parser.add_argument("--dataset", default="data/sift1m/sift1m.hdf5")
    parser.add_argument("--writer-url", default="http://127.0.0.1:8100")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    result = load_into_writer(
        args.dataset,
        args.writer_url.rstrip("/"),
        batch_size=args.batch_size,
        limit=args.limit,
        offset=args.offset,
        start_id=args.start_id,
        timeout_seconds=args.timeout,
    )
    print(
        "done "
        f"inserted={result['inserted']} "
        f"elapsed={result['elapsed_seconds']:.2f}s "
        f"rate={result['vectors_per_second']:.1f}/s"
    )


if __name__ == "__main__":
    main()
