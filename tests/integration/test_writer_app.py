"""Integration tests for the Phase 4 writer FastAPI app."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture
def writer_client(monkeypatch, tmp_path):
    monkeypatch.setenv("VECSCALE_NODE_ROLE", "writer")
    monkeypatch.setenv("VECSCALE_NODE_ID", "writer-0")
    monkeypatch.setenv("VECSCALE_WRITER_URL", "http://writer-0:8100")
    monkeypatch.setenv("VECSCALE_COORDINATOR_URLS", '["http://coordinator-0:8000"]')
    monkeypatch.setenv("VECSCALE_DIM", str(DIM))
    monkeypatch.setenv("VECSCALE_NLIST", str(NLIST))
    monkeypatch.setenv("VECSCALE_NPROBE", str(NLIST))
    monkeypatch.setenv("VECSCALE_SEGMENT_FLUSH_THRESHOLD", "3")
    monkeypatch.setenv("VECSCALE_SHARED_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("VECSCALE_WAL_PATH", str(tmp_path / "wal"))

    import vecscaledb.writer.app as app_mod

    app_mod = importlib.reload(app_mod)
    coordinator = FakeCoordinatorClient()
    pipeline = WriterPipeline(app_mod.settings, coordinator)
    app_mod.create_pipeline = lambda: pipeline

    with TestClient(app_mod.app) as client:
        yield client, pipeline, coordinator


def test_writer_insert_health_and_segment_registration(writer_client):
    client, pipeline, coordinator = writer_client
    vectors = np.eye(DIM, dtype=np.float32)[:3]

    resp = client.post(
        "/insert",
        json={"ids": [1, 2, 3], "vectors": vectors.tolist()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["inserted"] == 3
    assert body["segment_flushed"] is True
    assert body["ntotal"] == 3
    assert body["lsn"] == 1

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["ntotal"] == 3
    assert health["segments"] == 1
    assert health["wal_lsn"] == 1
    assert coordinator.nodes[0].role == "writer"
    assert len(coordinator.segments) == 1


def test_writer_rejects_bad_dimension(writer_client):
    client, _, _ = writer_client

    resp = client.post("/insert", json={"ids": [1], "vectors": [[1.0, 2.0]]})

    assert resp.status_code == 400
    assert "Expected vectors" in resp.json()["detail"]


def test_writer_delete_endpoint_hides_memtable_row(writer_client):
    client, pipeline, _ = writer_client
    vectors = np.eye(DIM, dtype=np.float32)[:2]
    client.post("/insert", json={"ids": [1, 2], "vectors": vectors.tolist()})

    resp = client.post("/delete", json={"ids": [1]})
    ids, _ = np.array([]), np.array([])

    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1

    import asyncio

    ids, _ = asyncio.run(pipeline.lsm_engine.search(vectors[0], top_k=10, nprobe=NLIST))
    assert 1 not in ids
