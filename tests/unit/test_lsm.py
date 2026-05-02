"""Unit tests for the Phase 2 LSM engine."""

import os

import numpy as np
import pytest

from vecscaledb.storage.lsm import LSMEngine, merge_results


DIM = 8
NLIST = 2


def make_engine(tmp_path, flush_threshold=10, merge_threshold=4):
    return LSMEngine(
        shared_storage=str(tmp_path / "storage"),
        wal_path=str(tmp_path / "wal"),
        shard_id=0,
        dim=DIM,
        nlist=NLIST,
        flush_threshold=flush_threshold,
        merge_threshold=merge_threshold,
    )


@pytest.mark.asyncio
async def test_wal_recovery_restores_memtable(tmp_path):
    rng = np.random.default_rng(21)
    vectors = rng.random((5, DIM), dtype=np.float32)
    ids = np.arange(5, dtype=np.int64)

    engine = make_engine(tmp_path, flush_threshold=100)
    await engine.start()
    await engine.insert(ids, vectors)
    engine.wal.close()

    recovered = make_engine(tmp_path, flush_threshold=100)
    await recovered.start()

    health = await recovered.health()
    assert health["memtable_size"] == 5
    result_ids, _ = await recovered.search(vectors[3], top_k=1, nprobe=NLIST)
    assert result_ids == [3]
    await recovered.stop()


@pytest.mark.asyncio
async def test_insert_above_threshold_flushes_segment(tmp_path):
    rng = np.random.default_rng(22)
    vectors = rng.random((6, DIM), dtype=np.float32)

    engine = make_engine(tmp_path, flush_threshold=5)
    await engine.start()
    result = await engine.insert(np.arange(6), vectors)

    health = await engine.health()
    assert result["segment_flushed"] is True
    assert health["segments"] == 1
    assert health["memtable_size"] == 0
    assert os.path.isfile(os.path.join(engine.segments[0].path, "vectors.npy"))
    await engine.stop()


@pytest.mark.asyncio
async def test_segment_merge_compacts_oldest_segments(tmp_path):
    rng = np.random.default_rng(23)
    engine = make_engine(tmp_path, flush_threshold=5, merge_threshold=4)
    await engine.start()

    for batch in range(4):
        start = batch * 5
        vectors = rng.random((5, DIM), dtype=np.float32)
        await engine.insert(np.arange(start, start + 5), vectors)

    health = await engine.health()
    assert health["segments"] == 1
    assert health["ntotal"] == 20
    segment_dirs = [
        entry
        for entry in os.listdir(engine.segments_path)
        if os.path.isdir(os.path.join(engine.segments_path, entry))
    ]
    assert len(segment_dirs) == 1
    await engine.stop()


def test_merge_results_deduplicates_and_keeps_smallest_distance():
    ids, distances = merge_results(
        [
            (np.array([3.0, 1.0]), np.array([1, 2])),
            (np.array([0.5, 9.0]), np.array([1, 3])),
        ],
        top_k=2,
    )

    assert ids == [1, 2]
    assert distances == [0.5, 1.0]
