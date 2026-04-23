"""Unit tests for IVFFlatIndex."""

import numpy as np
import pytest

from vecscaledb.index.ivf_flat import IVFFlatIndex


DIM = 128
NLIST = 16   # small for fast tests
N = 5000


@pytest.fixture
def random_vectors():
    rng = np.random.default_rng(42)
    return rng.random((N, DIM), dtype=np.float32)


@pytest.fixture
def trained_index(random_vectors):
    idx = IVFFlatIndex(DIM, NLIST)
    idx.train(random_vectors)
    return idx


class TestIVFFlatIndex:
    def test_train_and_add(self, trained_index, random_vectors):
        ids = np.arange(N, dtype=np.int64)
        trained_index.add(random_vectors, ids)
        assert trained_index.ntotal == N

    def test_add_before_train_raises(self, random_vectors):
        idx = IVFFlatIndex(DIM, NLIST)
        ids = np.arange(N, dtype=np.int64)
        with pytest.raises(RuntimeError, match="trained"):
            idx.add(random_vectors, ids)

    def test_search_returns_top_k(self, trained_index, random_vectors):
        ids = np.arange(N, dtype=np.int64)
        trained_index.add(random_vectors, ids)

        query = random_vectors[0]
        top_k = 10
        distances, result_ids = trained_index.search(query, top_k, nprobe=NLIST)

        assert len(result_ids) <= top_k
        assert len(distances) == len(result_ids)
        # The first result should be the query itself (distance ≈ 0)
        assert result_ids[0] == 0
        assert distances[0] < 1e-5

    def test_all_ids_in_range(self, trained_index, random_vectors):
        ids = np.arange(N, dtype=np.int64)
        trained_index.add(random_vectors, ids)

        query = random_vectors[42]
        _, result_ids = trained_index.search(query, top_k=50, nprobe=NLIST)

        for rid in result_ids:
            assert 0 <= rid < N

    def test_search_with_full_nprobe(self, trained_index, random_vectors):
        """With nprobe == nlist we should get exact brute-force results."""
        ids = np.arange(N, dtype=np.int64)
        trained_index.add(random_vectors, ids)

        query = random_vectors[100]
        distances, result_ids = trained_index.search(query, top_k=5, nprobe=NLIST)

        assert len(result_ids) == 5
        assert result_ids[0] == 100  # exact match

    def test_save_and_load(self, trained_index, random_vectors, tmp_path):
        ids = np.arange(N, dtype=np.int64)
        trained_index.add(random_vectors, ids)

        path = str(tmp_path / "test_index.faiss")
        trained_index.save(path)

        loaded = IVFFlatIndex(DIM, NLIST)
        loaded.load(path)

        assert loaded.ntotal == N

        # Search should return same results
        query = random_vectors[0]
        d1, i1 = trained_index.search(query, top_k=10, nprobe=NLIST)
        d2, i2 = loaded.search(query, top_k=10, nprobe=NLIST)
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_allclose(d1, d2, atol=1e-6)
