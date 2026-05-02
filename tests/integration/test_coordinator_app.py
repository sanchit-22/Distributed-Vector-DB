"""Integration tests for the Phase 3 coordinator app."""

import asyncio
import importlib

import pytest
from fastapi.testclient import TestClient

from vecscaledb.coordinator.etcd_client import EtcdClient
from vecscaledb.models import SegmentMeta


@pytest.fixture
def coordinator_client(monkeypatch):
    EtcdClient.reset_memory()
    monkeypatch.setenv("VECSCALE_NODE_ROLE", "coordinator")
    monkeypatch.setenv("VECSCALE_NODE_ID", "coordinator-0")
    monkeypatch.setenv("VECSCALE_ETCD_ENDPOINTS", '["memory://local"]')
    monkeypatch.setenv("VECSCALE_WRITER_URL", "http://writer-0:8100")

    import vecscaledb.coordinator.app as app_mod

    app_mod = importlib.reload(app_mod)
    with TestClient(app_mod.app) as client:
        yield client


def test_register_reader_appears_in_health_and_search_route(coordinator_client):
    resp = coordinator_client.post(
        "/register",
        json={
            "node_id": "reader-0",
            "role": "reader",
            "address": "http://reader-0:8200",
            "shard_ids": [0],
        },
    )
    assert resp.status_code == 200

    health = coordinator_client.get("/health").json()
    assert health["leader"] is True
    assert health["ring_size"] == 150
    assert any(node["node_id"] == "reader-0" for node in health["nodes"])

    route = coordinator_client.get("/route/search").json()
    assert route["readers"] == [
        {
            "node_id": "reader-0",
            "role": "reader",
            "address": "http://reader-0:8200",
            "shard_ids": [0],
        }
    ]


def test_insert_route_uses_registered_writer_and_reader_ring(coordinator_client):
    coordinator_client.post(
        "/register",
        json={
            "node_id": "writer-0",
            "role": "writer",
            "address": "http://writer-0:8100",
            "shard_ids": [],
        },
    )
    coordinator_client.post(
        "/register",
        json={
            "node_id": "reader-0",
            "role": "reader",
            "address": "http://reader-0:8200",
            "shard_ids": [0],
        },
    )

    route = coordinator_client.get("/route/insert", params={"count": 100}).json()

    assert route["writer_url"] == "http://writer-0:8100"
    assert route["writer_node_id"] == "writer-0"
    assert route["shard_owner"] == "reader-0"
    assert route["shard_id"] == 0


def test_segments_register_and_list_updates_snapshot(coordinator_client):
    segment = {
        "segment_id": "0000-0000000007-test",
        "shard_id": 0,
        "num_vectors": 10,
        "snapshot_id": 7,
        "path": "/shared_storage/segments/0000-0000000007-test",
    }

    resp = coordinator_client.post("/segments", json=segment)
    assert resp.status_code == 200
    assert resp.json()["snapshot_id"] == 7

    listed = coordinator_client.get("/segments").json()
    assert listed == [segment]

    deleted = coordinator_client.delete(f"/segments/{segment['segment_id']}")
    assert deleted.status_code == 200
    assert coordinator_client.get("/segments").json() == []


def test_query_hash_search_route_can_select_single_owner(coordinator_client):
    for index in range(3):
        coordinator_client.post(
            "/register",
            json={
                "node_id": f"reader-{index}",
                "role": "reader",
                "address": f"http://reader-{index}:820{index}",
                "shard_ids": [index],
            },
        )

    full_scan = coordinator_client.get("/route/search").json()["readers"]
    routed = coordinator_client.get("/route/search", params={"query_hash": 123}).json()[
        "readers"
    ]

    assert len(full_scan) == 3
    assert len(routed) == 1
    assert routed[0]["node_id"] in {node["node_id"] for node in full_scan}


def test_coordinator_search_merges_reader_results(coordinator_client, monkeypatch):
    import vecscaledb.coordinator.app as app_mod

    for index in range(2):
        coordinator_client.post(
            "/register",
            json={
                "node_id": f"reader-{index}",
                "role": "reader",
                "address": f"http://reader-{index}:8200",
                "shard_ids": [index],
            },
        )

    async def fake_search_reader(client, reader, req, snapshot_id):
        if reader.node_id == "reader-0":
            return {"ids": [1, 2], "distances": [0.5, 1.5]}
        return {"ids": [1, 3], "distances": [0.25, 0.75]}

    monkeypatch.setattr(app_mod, "_search_reader", fake_search_reader)

    resp = coordinator_client.post(
        "/search",
        json={"query": [0.0] * 128, "top_k": 2, "nprobe": 4},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ids": [1, 3], "distances": [0.25, 0.75], "incomplete": False}


def test_coordinator_search_marks_partial_reader_failure(coordinator_client, monkeypatch):
    import vecscaledb.coordinator.app as app_mod

    for index in range(2):
        coordinator_client.post(
            "/register",
            json={
                "node_id": f"reader-{index}",
                "role": "reader",
                "address": f"http://reader-{index}:8200",
                "shard_ids": [index],
            },
        )

    async def fake_search_reader(client, reader, req, snapshot_id):
        if reader.node_id == "reader-1":
            raise RuntimeError("reader unavailable")
        return {"ids": [7], "distances": [0.1]}

    monkeypatch.setattr(app_mod, "_search_reader", fake_search_reader)

    resp = coordinator_client.post(
        "/search",
        json={"query": [0.0] * 128, "top_k": 10, "nprobe": 4},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ids": [7], "distances": [0.10000000149011612], "incomplete": True}


def test_coordinator_replicated_search_round_robins_readers(
    coordinator_client, monkeypatch
):
    import vecscaledb.coordinator.app as app_mod

    app_mod.settings.search_fanout_mode = "replicated"
    app_mod.search_counter = 0
    for index in range(3):
        coordinator_client.post(
            "/register",
            json={
                "node_id": f"reader-{index}",
                "role": "reader",
                "address": f"http://reader-{index}:8200",
                "shard_ids": [index],
            },
        )

    async def fake_search_reader(client, reader, req, snapshot_id):
        return {
            "ids": [int(reader.node_id.removeprefix("reader-"))],
            "distances": [0.1],
        }

    monkeypatch.setattr(app_mod, "_search_reader", fake_search_reader)

    selected_ids = []
    for _ in range(4):
        resp = coordinator_client.post(
            "/search",
            json={"query": [0.0] * 128, "top_k": 1, "nprobe": 4},
        )
        assert resp.status_code == 200
        selected_ids.append(resp.json()["ids"][0])

    assert selected_ids == [0, 1, 2, 0]


def test_coordinator_search_falls_back_to_segment_snapshot(
    coordinator_client, monkeypatch
):
    import vecscaledb.coordinator.app as app_mod

    coordinator_client.post(
        "/register",
        json={
            "node_id": "reader-0",
            "role": "reader",
            "address": "http://reader-0:8200",
            "shard_ids": [0],
        },
    )
    segment = SegmentMeta(
        segment_id="0000-0000000042-test",
        shard_id=0,
        num_vectors=10,
        snapshot_id=42,
        path="/shared_storage/segments/0000-0000000042-test",
    )
    asyncio.run(
        app_mod.etcd.put(
            f"{app_mod.SEGMENT_PREFIX}{segment.segment_id}",
            segment.model_dump_json(),
        )
    )

    async def fake_search_reader(client, reader, req, snapshot_id):
        assert snapshot_id == 42
        return {"ids": [42], "distances": [0.0]}

    monkeypatch.setattr(app_mod, "_search_reader", fake_search_reader)

    resp = coordinator_client.post(
        "/search",
        json={"query": [0.0] * 128, "top_k": 1, "nprobe": 4},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ids": [42], "distances": [0.0], "incomplete": False}


def test_coordinator_health_clears_stale_local_leader_flag(coordinator_client):
    import vecscaledb.coordinator.app as app_mod

    app_mod.leader = True
    asyncio.run(app_mod.etcd.put("/vecscaledb/leader", "coordinator-1"))

    health = coordinator_client.get("/health").json()

    assert health["leader"] is False
