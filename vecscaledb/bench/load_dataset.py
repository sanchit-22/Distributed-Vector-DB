"""Dataset loading utilities for ANN benchmark HDF5 files."""

import h5py
import numpy as np


def load_sift1m(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the SIFT1M dataset from an HDF5 file.

    Returns:
        (train_vectors, query_vectors, ground_truth_neighbors)
        - train: [1_000_000, 128] float32
        - test:  [10_000, 128] float32
        - neighbors: [10_000, 100] int32  (ground truth k-NN IDs)
    """
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    return train, test, neighbors


def load_sift10k(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 10K-vector slice of SIFT1M for fast development testing.

    Returns:
        (train_vectors[10_000], query_vectors[100], ground_truth[100])
    """
    train, test, neighbors = load_sift1m(path)
    return train[:10_000], test[:100], neighbors[:100]
