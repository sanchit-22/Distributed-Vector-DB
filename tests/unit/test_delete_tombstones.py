"""Tests for delete tombstone persistence across flush and merge."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import numpy as np
import pytest

from vecscaledb.storage.lsm import LSMEngine


DIM = 8
NLIST = 2
FLUSH_THRESHOLD = 5
MERGE_THRESHOLD = 3


@pytest.fixture()
def engine_dirs():
    tmp = tempfile.mkdtemp()
    shared = os.path.join(tmp, "shared")
    wal = os.path.join(tmp, "wal")
    os.makedirs(shared, exist_ok=True)
    os.makedirs(wal, exist_ok=True)
    yield shared, wal
    shutil.rmtree(tmp, ignore_errors=True)


def _make_engine(shared: str, wal: str) -> LSMEngine:
    return LSMEngine(
        shared_storage=shared,
        wal_path=wal,
        shard_id=0,
        dim=DIM,
        nlist=NLIST,
        flush_threshold=FLUSH_THRESHOLD,
        merge_threshold=MERGE_THRESHOLD,
    )


def _random_vectors(n: int) -> np.ndarray:
    return np.random.default_rng(42).standard_normal((n, DIM)).astype(np.float32)


@pytest.mark.asyncio
async def test_delete_survives_flush(engine_dirs):
    """Delete a vector, flush the MemTable, then search — the deleted ID must stay hidden."""
    shared, wal = engine_dirs
    engine = _make_engine(shared, wal)
    await engine.start()

    # Insert 3 vectors
    ids = np.array([100, 200, 300], dtype=np.int64)
    vectors = _random_vectors(3)
    await engine.insert(ids, vectors)

    # Delete vector 200
    await engine.delete(np.array([200], dtype=np.int64))

    # Flush — this clears the MemTable including the delete marker
    await engine.flush()

    # Search — vector 200 must NOT appear
    result_ids, _ = await engine.search(vectors[1], top_k=10, nprobe=2)
    assert 200 not in result_ids, f"Deleted ID 200 reappeared after flush: {result_ids}"

    # Vectors 100 and 300 should still be found
    result_ids_all, _ = await engine.search(vectors[0], top_k=10, nprobe=2)
    assert 100 in result_ids_all

    await engine.stop()


@pytest.mark.asyncio
async def test_delete_survives_restart(engine_dirs):
    """Delete + flush + restart — the tombstone must persist on disk."""
    shared, wal = engine_dirs
    engine = _make_engine(shared, wal)
    await engine.start()

    ids = np.array([10, 20, 30], dtype=np.int64)
    vectors = _random_vectors(3)
    await engine.insert(ids, vectors)
    await engine.flush()

    # Delete after flush
    await engine.delete(np.array([20], dtype=np.int64))

    await engine.stop()

    # Restart
    engine2 = _make_engine(shared, wal)
    await engine2.start()

    result_ids, _ = await engine2.search(vectors[1], top_k=10, nprobe=2)
    assert 20 not in result_ids, f"Deleted ID 20 reappeared after restart: {result_ids}"

    await engine2.stop()


@pytest.mark.asyncio
async def test_tombstones_cleaned_during_merge(engine_dirs):
    """After merge, tombstoned IDs should be physically removed from the merged segment."""
    shared, wal = engine_dirs
    engine = LSMEngine(
        shared_storage=shared,
        wal_path=wal,
        shard_id=0,
        dim=DIM,
        nlist=NLIST,
        flush_threshold=1000,
        merge_threshold=100,  # High so auto-merge doesn't trigger
    )
    await engine.start()

    # Insert and flush 3 separate segments
    for batch in range(3):
        start_id = batch * 10
        ids = np.array([start_id, start_id + 1, start_id + 2], dtype=np.int64)
        vecs = _random_vectors(3)
        await engine.insert(ids, vecs)
        await engine.flush()

    assert len(engine.segments) == 3

    # Delete vector 0 (from first batch)
    await engine.delete(np.array([0], dtype=np.int64))

    # Lower merge threshold so manual merge actually processes the 3 segments
    engine.merge_threshold = 3

    # Manually trigger merge
    merged = await engine.merge()

    # After merge, vector 0 should be physically absent from the merged segment
    assert merged is not None, "Merge should have produced a segment"
    merged_ids, _ = merged.load_arrays()
    assert 0 not in merged_ids.tolist(), "Tombstoned ID 0 was not removed during merge"

    # Tombstone for ID 0 should be cleaned
    assert 0 not in engine._tombstones, "Tombstone was not cleaned after merge"

    await engine.stop()


@pytest.mark.asyncio
async def test_insert_after_delete_of_same_id(engine_dirs):
    """Re-inserting a deleted ID should make it visible again."""
    shared, wal = engine_dirs
    engine = _make_engine(shared, wal)
    await engine.start()

    ids = np.array([42], dtype=np.int64)
    vec = _random_vectors(1)
    await engine.insert(ids, vec)
    await engine.flush()

    # Delete
    await engine.delete(np.array([42], dtype=np.int64))

    # Re-insert with new vector
    new_vec = _random_vectors(1)
    await engine.insert(ids, new_vec)

    # Should be found (the re-insert should override the tombstone in the memtable)
    result_ids, _ = await engine.search(new_vec[0], top_k=10, nprobe=2)
    assert 42 in result_ids

    await engine.stop()
