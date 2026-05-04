"""Integration tests for reader crash recovery via shared storage."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import numpy as np
import pytest

from vecscaledb.index.segment import Segment
from vecscaledb.models import SegmentMeta
from vecscaledb.reader.search import SegmentCache


DIM = 8
NLIST = 2


@pytest.fixture()
def tmp_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _random_vectors(n: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def _create_test_segment(shard_id: int, snapshot_id: int, base_path: str) -> Segment:
    """Create a small test segment in shared storage."""
    ids = np.arange(shard_id * 100, shard_id * 100 + 20, dtype=np.int64)
    vectors = _random_vectors(20, seed=shard_id + snapshot_id)
    return Segment.create(
        shard_id=shard_id,
        snapshot_id=snapshot_id,
        vectors=vectors,
        ids=ids,
        dim=DIM,
        nlist=NLIST,
        base_path=os.path.join(base_path, "segments"),
    )


@pytest.mark.asyncio
async def test_reader_recovers_segments_after_restart(tmp_dir):
    """A reader that 'crashes' and restarts should re-load segments from shared storage."""
    # Create segments in shared storage
    seg = _create_test_segment(shard_id=0, snapshot_id=1, base_path=tmp_dir)
    all_metas = [seg.meta]

    # Simulate reader boot: create cache and refresh
    cache1 = SegmentCache(
        shared_storage=tmp_dir,
        shard_id=0,
        max_segments_in_memory=20,
        load_all_shards=True,
    )
    await cache1.refresh(all_metas)
    assert cache1.segments_loaded == 1

    # Search succeeds
    query = _random_vectors(1, seed=100)[0]
    ids_before, dists_before = cache1.search_all(query, top_k=5, nprobe=2, snapshot_id=999)
    assert len(ids_before) > 0

    # Simulate crash: discard the cache entirely
    del cache1

    # Simulate restart: new cache from scratch
    cache2 = SegmentCache(
        shared_storage=tmp_dir,
        shard_id=0,
        max_segments_in_memory=20,
        load_all_shards=True,
    )
    await cache2.refresh(all_metas)
    assert cache2.segments_loaded == 1

    # Search should give the same results
    ids_after, dists_after = cache2.search_all(query, top_k=5, nprobe=2, snapshot_id=999)
    assert ids_before == ids_after, "Results differ after reader restart"
    assert len(dists_after) > 0


@pytest.mark.asyncio
async def test_reader_handles_new_segments_after_recovery(tmp_dir):
    """After recovery, a reader should pick up new segments added while it was down."""
    # Start with one segment
    seg1 = _create_test_segment(shard_id=0, snapshot_id=1, base_path=tmp_dir)

    cache = SegmentCache(
        shared_storage=tmp_dir, shard_id=0,
        max_segments_in_memory=20, load_all_shards=True,
    )
    await cache.refresh([seg1.meta])
    assert cache.segments_loaded == 1
    initial_total = cache.ntotal

    # Simulate crash
    del cache

    # While reader is down, a new segment is created
    seg2 = _create_test_segment(shard_id=0, snapshot_id=2, base_path=tmp_dir)

    # Reader restarts and refreshes — should see both segments
    cache2 = SegmentCache(
        shared_storage=tmp_dir, shard_id=0,
        max_segments_in_memory=20, load_all_shards=True,
    )
    await cache2.refresh([seg1.meta, seg2.meta])
    assert cache2.segments_loaded == 2
    assert cache2.ntotal > initial_total


@pytest.mark.asyncio
async def test_reader_evicts_removed_segments_on_recovery(tmp_dir):
    """A restarted reader should not retain segments that were removed (e.g. after merge)."""
    seg1 = _create_test_segment(shard_id=0, snapshot_id=1, base_path=tmp_dir)
    seg2 = _create_test_segment(shard_id=0, snapshot_id=2, base_path=tmp_dir)

    cache = SegmentCache(
        shared_storage=tmp_dir, shard_id=0,
        max_segments_in_memory=20, load_all_shards=True,
    )
    await cache.refresh([seg1.meta, seg2.meta])
    assert cache.segments_loaded == 2

    # Simulate: after merge, only seg2 remains registered
    await cache.refresh([seg2.meta])
    assert cache.segments_loaded == 1
    assert cache.segment_ids == [seg2.segment_id]
