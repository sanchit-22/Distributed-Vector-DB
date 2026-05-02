"""Snapshot visibility tests."""

import numpy as np
import pytest

from vecscaledb.index.segment import Segment
from vecscaledb.storage.lsm import LSMEngine
from vecscaledb.storage.snapshot import SnapshotManager


DIM = 8
NLIST = 2


def test_visible_segments_filters_by_snapshot(tmp_path):
    vectors = np.random.default_rng(7).random((4, DIM), dtype=np.float32)
    first = Segment.create(0, 5, vectors[:2], np.array([1, 2]), DIM, NLIST, str(tmp_path))
    second = Segment.create(0, 10, vectors[2:], np.array([3, 4]), DIM, NLIST, str(tmp_path))

    manager = SnapshotManager()

    assert manager.visible_segments(5, [first, second]) == [first]
    assert manager.visible_segments(10, [first, second]) == [first, second]


@pytest.mark.asyncio
async def test_search_snapshot_hides_newer_memtable_writes(tmp_path):
    engine = LSMEngine(
        shared_storage=str(tmp_path / "storage"),
        wal_path=str(tmp_path / "wal"),
        shard_id=0,
        dim=DIM,
        nlist=NLIST,
        flush_threshold=100,
        merge_threshold=4,
    )
    await engine.start()

    rng = np.random.default_rng(11)
    first = rng.random((1, DIM), dtype=np.float32)
    second = rng.random((1, DIM), dtype=np.float32)

    result = await engine.insert(np.array([1]), first)
    snapshot_id = result["lsn"]
    await engine.insert(np.array([2]), second)

    ids, _ = await engine.search(second[0], top_k=10, nprobe=NLIST, snapshot_id=snapshot_id)
    assert 2 not in ids

    ids, _ = await engine.search(second[0], top_k=1, nprobe=NLIST)
    assert ids == [2]
    await engine.stop()
