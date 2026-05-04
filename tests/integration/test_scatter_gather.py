"""Integration tests for scatter-gather multi-shard search."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from vecscaledb.config import Settings
from vecscaledb.coordinator.etcd_client import EtcdClient
from vecscaledb.index.segment import Segment
from vecscaledb.models import NodeInfo, SegmentMeta
from vecscaledb.reader.search import SegmentCache
from vecscaledb.storage.lsm import merge_results


DIM = 8
NLIST = 2


@pytest.fixture()
def tmp_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _random_vectors(n: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def _create_segment(
    shard_id: int, snapshot_id: int, ids: np.ndarray, vectors: np.ndarray, base_path: str
) -> Segment:
    """Create and persist a segment for the given shard."""
    return Segment.create(
        shard_id=shard_id,
        snapshot_id=snapshot_id,
        vectors=vectors,
        ids=ids,
        dim=DIM,
        nlist=NLIST,
        base_path=base_path,
    )


def test_readers_load_only_own_shard(tmp_dir):
    """Each reader in scatter mode should load only segments matching its shard_id."""
    seg_path = os.path.join(tmp_dir, "segments")

    # Create segments for 3 different shards
    all_segments: list[SegmentMeta] = []
    for shard_id in range(3):
        ids = np.array([shard_id * 10 + j for j in range(5)], dtype=np.int64)
        vecs = _random_vectors(5, seed=shard_id)
        seg = _create_segment(shard_id, 1, ids, vecs, seg_path)
        all_segments.append(seg.meta)

    # Reader for shard 0 should only load shard 0's segment
    cache_0 = SegmentCache(
        shared_storage=tmp_dir,
        shard_id=0,
        max_segments_in_memory=20,
        load_all_shards=False,
    )
    import asyncio
    asyncio.get_event_loop().run_until_complete(cache_0.refresh(all_segments))
    assert cache_0.segments_loaded == 1

    # Reader for shard 1 should only load shard 1's segment
    cache_1 = SegmentCache(
        shared_storage=tmp_dir,
        shard_id=1,
        max_segments_in_memory=20,
        load_all_shards=False,
    )
    asyncio.get_event_loop().run_until_complete(cache_1.refresh(all_segments))
    assert cache_1.segments_loaded == 1

    # With load_all_shards=True, all 3 segments should load
    cache_all = SegmentCache(
        shared_storage=tmp_dir,
        shard_id=0,
        max_segments_in_memory=20,
        load_all_shards=True,
    )
    asyncio.get_event_loop().run_until_complete(cache_all.refresh(all_segments))
    assert cache_all.segments_loaded == 3


def test_scatter_gather_merge_finds_all_vectors(tmp_dir):
    """Simulated scatter-gather: each reader searches its shard, coordinator merges."""
    seg_path = os.path.join(tmp_dir, "segments")

    # Create vectors spread across 2 shards
    all_vectors = _random_vectors(10, seed=99)
    all_ids = np.arange(10, dtype=np.int64)

    # Shard 0 gets even IDs, shard 1 gets odd IDs
    shard_0_mask = all_ids % 2 == 0
    shard_1_mask = all_ids % 2 == 1

    seg0 = _create_segment(
        0, 1,
        all_ids[shard_0_mask], all_vectors[shard_0_mask], seg_path,
    )
    seg1 = _create_segment(
        1, 1,
        all_ids[shard_1_mask], all_vectors[shard_1_mask], seg_path,
    )

    all_metas = [seg0.meta, seg1.meta]

    # Reader 0 searches shard 0
    cache_0 = SegmentCache(tmp_dir, shard_id=0, load_all_shards=False)
    import asyncio
    asyncio.get_event_loop().run_until_complete(cache_0.refresh(all_metas))

    # Reader 1 searches shard 1
    cache_1 = SegmentCache(tmp_dir, shard_id=1, load_all_shards=False)
    asyncio.get_event_loop().run_until_complete(cache_1.refresh(all_metas))

    # Pick query = first vector (ID=0, in shard 0)
    query = all_vectors[0]
    top_k = 10

    ids_0, dists_0 = cache_0.search_all(query, top_k=top_k, nprobe=2, snapshot_id=999)
    ids_1, dists_1 = cache_1.search_all(query, top_k=top_k, nprobe=2, snapshot_id=999)

    # Merge results (simulating coordinator)
    results = []
    if ids_0:
        results.append((np.array(dists_0, dtype=np.float32), np.array(ids_0, dtype=np.int64)))
    if ids_1:
        results.append((np.array(dists_1, dtype=np.float32), np.array(ids_1, dtype=np.int64)))

    merged_ids, merged_dists = merge_results(results, top_k)

    # The query is vector 0 — it should be found (distance ≈ 0)
    assert 0 in merged_ids, f"Query vector ID 0 not found in merged results: {merged_ids}"
    # Should have results from both shards
    assert len(merged_ids) == 10, f"Expected 10 results, got {len(merged_ids)}"

    # Verify results from both shards are present
    has_even = any(vid % 2 == 0 for vid in merged_ids)
    has_odd = any(vid % 2 == 1 for vid in merged_ids)
    assert has_even, "No results from shard 0 (even IDs)"
    assert has_odd, "No results from shard 1 (odd IDs)"
