"""Tests for the coordinator metadata client memory backend."""

import pytest

from vecscaledb.coordinator.etcd_client import EtcdClient


@pytest.fixture(autouse=True)
def reset_memory_etcd():
    EtcdClient.reset_memory()


@pytest.mark.asyncio
async def test_memory_put_get_prefix_delete():
    client = EtcdClient(["memory://local"])

    await client.put("/vecscaledb/nodes/reader-0", "a")
    await client.put("/vecscaledb/nodes/reader-1", "b")
    await client.put("/other", "c")

    assert await client.get("/vecscaledb/nodes/reader-0") == "a"
    assert await client.get_prefix("/vecscaledb/nodes/") == {
        "/vecscaledb/nodes/reader-0": "a",
        "/vecscaledb/nodes/reader-1": "b",
    }

    await client.delete("/vecscaledb/nodes/reader-0")
    assert await client.get("/vecscaledb/nodes/reader-0") is None


@pytest.mark.asyncio
async def test_memory_campaign_picks_single_leader():
    first = EtcdClient(["memory://local"])
    second = EtcdClient(["memory://local"])

    assert await first.campaign("coordinator", "coordinator-0") is True
    assert await second.campaign("coordinator", "coordinator-1") is False
    assert await second.get("/vecscaledb/leader") == "coordinator-0"
