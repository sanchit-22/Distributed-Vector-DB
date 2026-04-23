"""Recall@k evaluation script for VecScaleDB.

Usage (against a running single-node server):
    python -m vecscaledb.bench.recall_eval [--host HOST] [--port PORT] [--dataset PATH]
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx
import numpy as np

from vecscaledb.bench.load_dataset import load_sift10k


def compute_recall_at_k(
    predicted_ids: np.ndarray, ground_truth: np.ndarray, k: int
) -> float:
    """Compute mean recall@k across all queries.

    Args:
        predicted_ids: ``[N, k]`` — top-k result IDs for N queries.
        ground_truth:  ``[N, R]`` — true nearest neighbours (R >= k).
        k: number of neighbours to evaluate.

    Returns:
        Mean recall@k ∈ [0, 1].
    """
    n = predicted_ids.shape[0]
    recalls = []
    for i in range(n):
        gt_set = set(ground_truth[i, :k].tolist())
        pred_set = set(predicted_ids[i, :k].tolist())
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall evaluation for VecScaleDB")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset",
        default="data/sift1m/sift1m.hdf5",
        help="Path to SIFT HDF5 file",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # 1. Load SIFT10K
    print("Loading SIFT10K …")
    train, queries, ground_truth = load_sift10k(args.dataset)
    n_train = train.shape[0]
    print(f"  train: {train.shape}, queries: {queries.shape}, gt: {ground_truth.shape}")

    # 2. Insert in batches of 1000
    batch_size = 1000
    print(f"Inserting {n_train} vectors in batches of {batch_size} …")
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            ids = list(range(start, end))
            vectors = train[start:end].tolist()
            resp = client.post("/insert", json={"ids": ids, "vectors": vectors})
            resp.raise_for_status()
        print(f"  Inserted {n_train} vectors.")

        # Check health
        health = client.get("/health").json()
        print(f"  Health: {health}")

        # 3. Query
        print("Running search queries …")
        n_queries = queries.shape[0]
        top_k = 100
        all_pred_ids = []

        t0 = time.perf_counter()
        for i in range(n_queries):
            resp = client.post(
                "/search",
                json={"query": queries[i].tolist(), "top_k": top_k, "nprobe": 32},
            )
            resp.raise_for_status()
            result = resp.json()
            pred = result["ids"]
            # Pad with -1 if fewer than top_k results
            pred += [-1] * (top_k - len(pred))
            all_pred_ids.append(pred[:top_k])
        elapsed = time.perf_counter() - t0

        predicted = np.array(all_pred_ids, dtype=np.int64)
        print(f"  {n_queries} queries in {elapsed:.2f}s ({n_queries/elapsed:.1f} QPS)")

    # 4. Compute recall
    recall_10 = compute_recall_at_k(predicted, ground_truth, k=10)
    recall_100 = compute_recall_at_k(predicted, ground_truth, k=100)

    print(f"\nRecall@10  = {recall_10:.4f}")
    print(f"Recall@100 = {recall_100:.4f}")

    if recall_10 >= 0.95:
        print("✓ recall@10 >= 0.95 — PASS")
    else:
        print("✗ recall@10 < 0.95 — FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
