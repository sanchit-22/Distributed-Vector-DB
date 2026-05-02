"""Integration tests for the Phase 5 reader FastAPI app."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vecscaledb.index.segment import Segment
from vecscaledb.models import NodeInfo, SegmentMeta
from vecscaledb.reader.search import SegmentCache


DIM = 8
NLIST = 2


class FakeCoordinatorClient:
    def __init__(self, segments: list[SegmentMeta]) -> None:
        self.nodes: list[NodeInfo] = []
        self.segments = segments

    async def register_node(self, info: NodeInfo) -> None:
        self.nodes.append(info)

    async def get_all_segments(self) -> list[SegmentMeta]:
        return list(self.segments)


@pytest.fixture
def reader_client(monkeypatch, tmp_path):
    vectors = np.eye(DIM, dtype=np.float32)[:2]
    segment = Segment.create(
        shard_id=0,
        snapshot_id=7,
        vectors=vectors,
        ids=np.array([10, 11], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "segments"),
    )

    monkeypatch.setenv("VECSCALE_NODE_ROLE", "reader")
    monkeypatch.setenv("VECSCALE_NODE_ID", "reader-0")
    monkeypatch.setenv("VECSCALE_READER_SHARD_ID", "0")
    monkeypatch.setenv("VECSCALE_DIM", str(DIM))
    monkeypatch.setenv("VECSCALE_NLIST", str(NLIST))
    monkeypatch.setenv("VECSCALE_NPROBE", str(NLIST))
    monkeypatch.setenv("VECSCALE_SHARED_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("VECSCALE_COORDINATOR_URLS", '["http://coordinator-0:8000"]')

    import vecscaledb.reader.app as app_mod

    app_mod = importlib.reload(app_mod)
    coordinator = FakeCoordinatorClient([segment.meta])
    app_mod.create_coordinator_client = lambda: coordinator
    app_mod.create_cache = lambda: SegmentCache(str(tmp_path / "storage"), shard_id=0)

    with TestClient(app_mod.app) as client:
        yield client, coordinator, vectors


def test_reader_start_registers_and_loads_segments(reader_client):
    client, coordinator, _ = reader_client

    health = client.get("/health").json()
    ready = client.get("/ready").json()

    assert coordinator.nodes[0].role == "reader"
    assert health["status"] == "ok"
    assert health["segments_loaded"] == 1
    assert health["ntotal"] == 2
    assert ready["status"] == "ready"
    assert ready["segments_loaded"] == 1


def test_reader_ready_returns_503_until_first_refresh(reader_client):
    client, _, _ = reader_client

    import vecscaledb.reader.app as app_mod

    app_mod.first_refresh_complete = False
    resp = client.get("/ready")

    assert resp.status_code == 503
    assert "Initial segment refresh" in resp.json()["detail"]
    app_mod.first_refresh_complete = True


def test_reader_search_respects_snapshot(reader_client):
    client, _, vectors = reader_client

    old = client.post(
        "/search",
        json={"query": vectors[1].tolist(), "top_k": 1, "nprobe": NLIST, "snapshot_id": 6},
    )
    current = client.post(
        "/search",
        json={"query": vectors[1].tolist(), "top_k": 1, "nprobe": NLIST, "snapshot_id": 7},
    )

    assert old.json()["ids"] == []
    assert current.json()["ids"] == [11]


def test_reader_rejects_bad_query_dimension(reader_client):
    client, _, _ = reader_client

    resp = client.post(
        "/search",
        json={"query": [1.0, 2.0], "top_k": 1, "nprobe": NLIST, "snapshot_id": 7},
    )

    assert resp.status_code == 400
