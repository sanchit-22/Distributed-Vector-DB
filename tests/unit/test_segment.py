"""Unit tests for the Segment abstraction."""

import numpy as np
import pytest

from vecscaledb.index.segment import Segment


DIM = 128
NLIST = 16
N = 2000


@pytest.fixture
def random_data():
    rng = np.random.default_rng(123)
    vectors = rng.random((N, DIM), dtype=np.float32)
    ids = np.arange(N, dtype=np.int64)
    return vectors, ids


class TestSegment:
    def test_create_writes_files(self, random_data, tmp_path):
        vectors, ids = random_data
        seg = Segment.create(
            shard_id=0,
            snapshot_id=0,
            vectors=vectors,
            ids=ids,
            dim=DIM,
            nlist=NLIST,
            base_path=str(tmp_path),
        )

        # Check files exist
        import os

        assert os.path.isfile(os.path.join(seg.path, "index.faiss"))
        assert os.path.isfile(os.path.join(seg.path, "ids.npy"))
        assert os.path.isfile(os.path.join(seg.path, "meta.json"))
        assert seg.ntotal == N

    def test_load_restores_segment(self, random_data, tmp_path):
        vectors, ids = random_data
        seg = Segment.create(
            shard_id=0,
            snapshot_id=1,
            vectors=vectors,
            ids=ids,
            dim=DIM,
            nlist=NLIST,
            base_path=str(tmp_path),
        )

        loaded = Segment.load(seg.path)

        assert loaded.segment_id == seg.segment_id
        assert loaded.ntotal == N
        assert loaded.meta.num_vectors == N

    def test_search_consistency_after_reload(self, random_data, tmp_path):
        vectors, ids = random_data
        seg = Segment.create(
            shard_id=0,
            snapshot_id=2,
            vectors=vectors,
            ids=ids,
            dim=DIM,
            nlist=NLIST,
            base_path=str(tmp_path),
        )

        loaded = Segment.load(seg.path)

        query = vectors[42]
        d1, i1 = seg.search(query, top_k=10, nprobe=NLIST)
        d2, i2 = loaded.search(query, top_k=10, nprobe=NLIST)

        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_allclose(d1, d2, atol=1e-6)

    def test_meta_fields(self, random_data, tmp_path):
        vectors, ids = random_data
        seg = Segment.create(
            shard_id=3,
            snapshot_id=7,
            vectors=vectors,
            ids=ids,
            dim=DIM,
            nlist=NLIST,
            base_path=str(tmp_path),
        )

        assert seg.shard_id == 3
        assert seg.snapshot_id == 7
        assert seg.meta.shard_id == 3
        assert seg.meta.snapshot_id == 7
        assert seg.meta.num_vectors == N
