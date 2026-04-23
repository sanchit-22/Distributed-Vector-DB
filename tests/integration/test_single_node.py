"""Integration test: single-node insert + search via the REST API.

Uses ASGI TestClient so no external server is needed.
Tests with random data (SIFT10K requires the dataset file).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient


DIM = 128
N = 5000
NLIST = 16


@pytest.fixture(autouse=True)
def reset_node_state(monkeypatch, tmp_path):
    """Reset global state and use a temp directory for shared storage."""
    monkeypatch.setenv("VECSCALE_DIM", str(DIM))
    monkeypatch.setenv("VECSCALE_NLIST", str(NLIST))
    monkeypatch.setenv("VECSCALE_NPROBE", str(NLIST))
    monkeypatch.setenv("VECSCALE_SEGMENT_FLUSH_THRESHOLD", str(N + 1))  # no auto-flush
    monkeypatch.setenv("VECSCALE_SHARED_STORAGE_PATH", str(tmp_path / "storage"))

    # Re-import to pick up monkeypatched env
    import importlib
    import vecscaledb.node as node_mod

    # Reset module-level state
    node_mod._memtable.clear()
    node_mod._segments.clear()
    node_mod._next_snapshot_id = 0

    # Reload settings
    from vecscaledb.config import Settings
    node_mod.settings = Settings()

    yield node_mod


@pytest.fixture
def client(reset_node_state):
    from vecscaledb.node import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def random_data():
    rng = np.random.default_rng(99)
    vectors = rng.random((N, DIM), dtype=np.float32)
    ids = list(range(N))
    return vectors, ids


class TestSingleNode:
    def test_health_empty(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ntotal"] == 0

    def test_insert_and_health(self, client, random_data):
        vectors, ids = random_data
        batch_size = 1000
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            resp = client.post(
                "/insert",
                json={
                    "ids": ids[start:end],
                    "vectors": vectors[start:end].tolist(),
                },
            )
            assert resp.status_code == 200

        health = client.get("/health").json()
        assert health["ntotal"] == N

    def test_search_returns_results(self, client, random_data):
        vectors, ids = random_data
        # Insert all
        client.post(
            "/insert",
            json={"ids": ids, "vectors": vectors.tolist()},
        )

        # Search for vector 0 — should find itself
        resp = client.post(
            "/search",
            json={"query": vectors[0].tolist(), "top_k": 10, "nprobe": NLIST},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert 0 in result["ids"]
        assert len(result["ids"]) <= 10

    def test_search_exact_match(self, client, random_data):
        vectors, ids = random_data
        client.post(
            "/insert",
            json={"ids": ids, "vectors": vectors.tolist()},
        )

        resp = client.post(
            "/search",
            json={"query": vectors[42].tolist(), "top_k": 1, "nprobe": NLIST},
        )
        result = resp.json()
        assert result["ids"][0] == 42
        assert result["distances"][0] < 1e-5

    def test_insert_triggers_flush(self, client, random_data, reset_node_state):
        """Insert more than flush threshold → should create a segment."""
        reset_node_state.settings.segment_flush_threshold = 1000
        vectors, ids = random_data

        # Insert 2000 vectors (above threshold of 1000)
        for start in range(0, 2000, 500):
            end = start + 500
            resp = client.post(
                "/insert",
                json={"ids": ids[start:end], "vectors": vectors[start:end].tolist()},
            )
            assert resp.status_code == 200

        health = client.get("/health").json()
        assert health["ntotal"] >= 2000
