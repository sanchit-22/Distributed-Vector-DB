"""Tests for reader-side segment caching and search."""

import numpy as np
import pytest

from vecscaledb.index.segment import Segment
from vecscaledb.reader.search import SegmentCache


DIM = 8
NLIST = 2


def create_segment(tmp_path, snapshot_id, ids, vectors, shard_id=0):
    return Segment.create(
        shard_id=shard_id,
        snapshot_id=snapshot_id,
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        ids=np.array(ids, dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_segment_cache_refresh_loads_and_searches_visible_segments(tmp_path):
    vectors = np.eye(DIM, dtype=np.float32)[:2]
    first = create_segment(tmp_path, 5, [1], vectors[:1])
    second = create_segment(tmp_path, 10, [2], vectors[1:2])

    cache = SegmentCache(str(tmp_path), shard_id=0)
    await cache.refresh([first.meta, second.meta])

    old_ids, _ = cache.search_all(vectors[1], top_k=10, nprobe=NLIST, snapshot_id=5)
    current_ids, _ = cache.search_all(vectors[1], top_k=1, nprobe=NLIST, snapshot_id=10)

    assert cache.segments_loaded == 2
    assert 2 not in old_ids
    assert current_ids == [2]


@pytest.mark.asyncio
async def test_segment_cache_filters_shard_and_evicts_removed_segments(tmp_path):
    vectors = np.eye(DIM, dtype=np.float32)[:3]
    own = create_segment(tmp_path, 1, [1], vectors[:1], shard_id=0)
    other = create_segment(tmp_path, 2, [2], vectors[1:2], shard_id=1)

    cache = SegmentCache(str(tmp_path), shard_id=0)
    await cache.refresh([own.meta, other.meta])
    assert cache.segment_ids == [own.segment_id]

    await cache.refresh([])
    assert cache.segments_loaded == 0


@pytest.mark.asyncio
async def test_segment_cache_can_load_all_shards_for_replicated_readers(tmp_path):
    vectors = np.eye(DIM, dtype=np.float32)[:2]
    segment_a = Segment.create(
        shard_id=0,
        snapshot_id=1,
        vectors=vectors,
        ids=np.array([10, 11], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "segments"),
    )
    segment_b = Segment.create(
        shard_id=4,
        snapshot_id=2,
        vectors=vectors,
        ids=np.array([20, 21], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "segments"),
    )
    cache = SegmentCache(str(tmp_path), shard_id=2, load_all_shards=True)

    await cache.refresh([segment_a.meta, segment_b.meta])

    assert cache.segments_loaded == 2
    assert set(cache.segment_ids) == {segment_a.segment_id, segment_b.segment_id}


@pytest.mark.asyncio
async def test_segment_cache_lru_limit(tmp_path):
    vectors = np.eye(DIM, dtype=np.float32)[:3]
    segments = [
        create_segment(tmp_path, index + 1, [index], vectors[index : index + 1])
        for index in range(3)
    ]

    cache = SegmentCache(str(tmp_path), shard_id=0, max_segments_in_memory=2)
    await cache.refresh([segment.meta for segment in segments])

    assert cache.segments_loaded == 2
    assert segments[0].segment_id not in cache.segment_ids
