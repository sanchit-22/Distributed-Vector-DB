"""Tests for the Phase 4 writer pipeline."""

import numpy as np
import pytest

from vecscaledb.config import Settings
from vecscaledb.models import NodeInfo, SegmentMeta
from vecscaledb.writer.pipeline import WriterPipeline


DIM = 8
NLIST = 2


class FakeCoordinatorClient:
    def __init__(self) -> None:
        self.nodes: list[NodeInfo] = []
        self.segments: list[SegmentMeta] = []

    async def register_node(self, info: NodeInfo) -> None:
        self.nodes.append(info)

    async def register_segment(self, meta: SegmentMeta) -> None:
        if meta.segment_id not in {seg.segment_id for seg in self.segments}:
            self.segments.append(meta)

    async def delete_segment(self, segment_id: str) -> None:
        self.segments = [seg for seg in self.segments if seg.segment_id != segment_id]

    async def get_all_segments(self) -> list[SegmentMeta]:
        return list(self.segments)


def make_settings(tmp_path, flush_threshold=100):
    return Settings(
        node_role="writer",
        node_id="writer-0",
        writer_url="http://writer-0:8100",
        coordinator_urls=["http://coordinator-0:8000"],
        shared_storage_path=str(tmp_path / "storage"),
        wal_path=str(tmp_path / "wal"),
        dim=DIM,
        nlist=NLIST,
        nprobe=NLIST,
        segment_flush_threshold=flush_threshold,
        segment_merge_threshold=4,
    )


@pytest.mark.asyncio
async def test_writer_start_registers_node(tmp_path):
    coordinator = FakeCoordinatorClient()
    pipeline = WriterPipeline(make_settings(tmp_path), coordinator)

    await pipeline.start()

    assert coordinator.nodes[0].node_id == "writer-0"
    assert coordinator.nodes[0].role == "writer"
    await pipeline.stop()


@pytest.mark.asyncio
async def test_writer_can_refresh_node_registration(tmp_path):
    coordinator = FakeCoordinatorClient()
    pipeline = WriterPipeline(make_settings(tmp_path), coordinator)

    await pipeline.start()
    await pipeline.register_self()

    assert [node.node_id for node in coordinator.nodes] == ["writer-0", "writer-0"]
    await pipeline.stop()


@pytest.mark.asyncio
async def test_writer_insert_registers_flushed_segment(tmp_path):
    coordinator = FakeCoordinatorClient()
    pipeline = WriterPipeline(make_settings(tmp_path, flush_threshold=3), coordinator)
    await pipeline.start()

    vectors = np.eye(DIM, dtype=np.float32)[:3]
    result = await pipeline.insert(np.array([1, 2, 3]), vectors)

    assert result["inserted"] == 3
    assert result["segment_flushed"] is True
    assert result["ntotal"] == 3
    assert len(coordinator.segments) == 1
    assert coordinator.segments[0].num_vectors == 3
    await pipeline.stop()


@pytest.mark.asyncio
async def test_writer_recovery_registers_unregistered_existing_segment(tmp_path):
    settings = make_settings(tmp_path, flush_threshold=2)
    first_coordinator = FakeCoordinatorClient()
    first = WriterPipeline(settings, first_coordinator)
    await first.start()
    vectors = np.eye(DIM, dtype=np.float32)[:2]
    await first.lsm_engine.insert(np.array([10, 11]), vectors)
    first.lsm_engine.wal.close()

    second_coordinator = FakeCoordinatorClient()
    recovered = WriterPipeline(settings, second_coordinator)
    await recovered.start()

    assert len(second_coordinator.segments) == 1
    assert second_coordinator.segments[0].num_vectors == 2
    await recovered.stop()


@pytest.mark.asyncio
async def test_writer_wal_recovery_preserves_unflushed_acknowledged_writes(tmp_path):
    settings = make_settings(tmp_path, flush_threshold=5)
    coordinator = FakeCoordinatorClient()
    first = WriterPipeline(settings, coordinator)
    await first.start()

    initial_vectors = np.eye(DIM, dtype=np.float32)[:3]
    result = await first.insert(np.array([100, 101, 102]), initial_vectors)
    assert result["segment_flushed"] is False
    assert result["wal_lsn"] == 1
    # Simulate a crash: close the WAL file without graceful pipeline shutdown,
    # because graceful shutdown intentionally flushes the MemTable.
    first.lsm_engine.wal.close()
    first._started = False

    recovered = WriterPipeline(settings, coordinator)
    await recovered.start()
    recovered_ids, _ = await recovered.lsm_engine.search(
        initial_vectors[1], top_k=5, nprobe=NLIST
    )
    assert 101 in recovered_ids

    more_vectors = np.eye(DIM, dtype=np.float32)[3:5]
    flush_result = await recovered.insert(np.array([103, 104]), more_vectors)

    assert flush_result["segment_flushed"] is True
    assert len(coordinator.segments) == 1
    assert coordinator.segments[0].num_vectors == 5
    for vector_id, query in zip([100, 101, 102, 103, 104], np.eye(DIM, dtype=np.float32)[:5]):
        ids, _ = await recovered.lsm_engine.search(query, top_k=5, nprobe=NLIST)
        assert vector_id in ids
    await recovered.stop()


@pytest.mark.asyncio
async def test_writer_delete_hides_id_from_engine_search(tmp_path):
    coordinator = FakeCoordinatorClient()
    pipeline = WriterPipeline(make_settings(tmp_path, flush_threshold=100), coordinator)
    await pipeline.start()

    vectors = np.eye(DIM, dtype=np.float32)[:2]
    await pipeline.insert(np.array([1, 2]), vectors)
    result = await pipeline.delete(np.array([1]))
    ids, _ = await pipeline.lsm_engine.search(vectors[0], top_k=10, nprobe=NLIST)

    assert result["deleted"] == 1
    assert 1 not in ids
    await pipeline.stop()


@pytest.mark.asyncio
async def test_writer_reconciles_compacted_segments(tmp_path):
    coordinator = FakeCoordinatorClient()
    pipeline = WriterPipeline(make_settings(tmp_path, flush_threshold=2), coordinator)
    await pipeline.start()

    for batch in range(4):
        ids = np.array([batch * 2, batch * 2 + 1])
        vectors = np.zeros((2, DIM), dtype=np.float32)
        vectors[0, batch] = 1.0
        vectors[1, batch + 4] = 1.0
        await pipeline.insert(ids, vectors)

    local_ids = {segment.segment_id for segment in pipeline.lsm_engine.segments}
    remote_ids = {segment.segment_id for segment in coordinator.segments}

    assert len(local_ids) == 1
    assert remote_ids == local_ids
    await pipeline.stop()
