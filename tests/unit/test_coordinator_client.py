"""Tests for the coordinator HTTP failover client."""

import httpx
import pytest

from vecscaledb.coordinator.client import CoordinatorClient
from vecscaledb.models import NodeInfo


@pytest.mark.asyncio
async def test_coordinator_client_retries_next_url(monkeypatch):
    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, **kwargs):
            calls.append(url)
            if "coordinator-0" in url:
                raise httpx.ConnectError("down", request=httpx.Request(method, url))
            return httpx.Response(
                200,
                json={"registered": True},
                request=httpx.Request(method, url),
            )

    monkeypatch.setattr("vecscaledb.coordinator.client.httpx.AsyncClient", FakeAsyncClient)
    client = CoordinatorClient(
        ["http://coordinator-0:8000", "http://coordinator-1:8000"],
        timeout=0.1,
    )

    await client.register_node(
        NodeInfo(
            node_id="reader-0",
            role="reader",
            address="http://reader-0:8200",
            shard_ids=[0],
        )
    )

    assert calls == [
        "http://coordinator-0:8000/register",
        "http://coordinator-1:8000/register",
    ]


@pytest.mark.asyncio
async def test_coordinator_client_retries_after_5xx(monkeypatch):
    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, **kwargs):
            calls.append(url)
            if "coordinator-0" in url:
                return httpx.Response(
                    503,
                    json={"detail": "No active coordinator leader."},
                    request=httpx.Request(method, url),
                )
            return httpx.Response(
                200,
                json=[],
                request=httpx.Request(method, url),
            )

    monkeypatch.setattr("vecscaledb.coordinator.client.httpx.AsyncClient", FakeAsyncClient)
    client = CoordinatorClient(
        ["http://coordinator-0:8000", "http://coordinator-1:8000"],
        timeout=0.1,
    )

    segments = await client.get_all_segments()

    assert segments == []
    assert calls == [
        "http://coordinator-0:8000/segments",
        "http://coordinator-1:8000/segments",
    ]
