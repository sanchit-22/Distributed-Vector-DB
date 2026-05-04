"""Integration tests for coordinator HA leader failover."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vecscaledb.coordinator.etcd_client import EtcdClient, _GLOBAL_MEMORY_STORE


@pytest.fixture(autouse=True)
def _reset_etcd():
    EtcdClient.reset_memory()
    yield
    EtcdClient.reset_memory()


@pytest.mark.asyncio
async def test_ttl_expiry_removes_key():
    """Keys with a TTL should disappear after the TTL expires."""
    client = EtcdClient(["memory://local"])

    await client.put("/test/key", "value", ttl=1)
    assert await client.get("/test/key") == "value"

    # Simulate TTL expiry by manipulating the monotonic clock offset
    _GLOBAL_MEMORY_STORE.ttls["/test/key"] = time.monotonic() - 1
    assert await client.get("/test/key") is None


@pytest.mark.asyncio
async def test_leader_failover_on_ttl_expiry():
    """When the leader's election key expires, a new coordinator should win the election."""
    # Coordinator A wins the initial election
    client_a = EtcdClient(["memory://local"])
    is_leader_a = await client_a.campaign("coordinator", "coordinator-a")
    assert is_leader_a is True

    # Coordinator B should not be leader
    client_b = EtcdClient(["memory://local"])
    is_leader_b = await client_b.campaign("coordinator", "coordinator-b")
    assert is_leader_b is False

    # Verify A is still leader
    leader = await client_a.get("/vecscaledb/leader")
    assert leader == "coordinator-a"

    # Simulate A crashing: its election key TTL expires
    election_key = "/vecscaledb/elections/coordinator"
    _GLOBAL_MEMORY_STORE.ttls[election_key] = time.monotonic() - 1
    _GLOBAL_MEMORY_STORE.ttls["/vecscaledb/leader"] = time.monotonic() - 1

    # Now B campaigns again and should win
    is_leader_b_now = await client_b.campaign("coordinator", "coordinator-b")
    assert is_leader_b_now is True

    leader_now = await client_b.get("/vecscaledb/leader")
    assert leader_now == "coordinator-b"


@pytest.mark.asyncio
async def test_leader_refreshes_ttl():
    """The current leader should be able to refresh its TTL by re-campaigning."""
    client = EtcdClient(["memory://local"])

    # Win the election
    assert await client.campaign("coordinator", "node-0") is True

    # Record the initial expiry
    election_key = "/vecscaledb/elections/coordinator"
    initial_expiry = _GLOBAL_MEMORY_STORE.ttls.get(election_key, 0)

    # Re-campaign (heartbeat) should refresh the TTL
    await asyncio.sleep(0.01)
    assert await client.campaign("coordinator", "node-0") is True
    refreshed_expiry = _GLOBAL_MEMORY_STORE.ttls.get(election_key, 0)

    assert refreshed_expiry > initial_expiry, "TTL was not refreshed by re-campaigning"


@pytest.mark.asyncio
async def test_node_registration_ttl_expiry():
    """Node registrations with TTL should expire, simulating node death."""
    client = EtcdClient(["memory://local"])

    # Register a reader with TTL
    await client.put("/vecscaledb/nodes/reader-0", '{"node_id":"reader-0"}', ttl=10)
    nodes = await client.get_prefix("/vecscaledb/nodes/")
    assert "reader-0" in str(nodes)

    # Simulate reader crash by expiring its key
    _GLOBAL_MEMORY_STORE.ttls["/vecscaledb/nodes/reader-0"] = time.monotonic() - 1

    nodes_after = await client.get_prefix("/vecscaledb/nodes/")
    assert "/vecscaledb/nodes/reader-0" not in nodes_after


@pytest.mark.asyncio
async def test_get_prefix_prunes_expired():
    """get_prefix should not return expired keys."""
    client = EtcdClient(["memory://local"])

    await client.put("/prefix/alive", "yes", ttl=100)
    await client.put("/prefix/dead", "no", ttl=1)

    # Expire the dead key
    _GLOBAL_MEMORY_STORE.ttls["/prefix/dead"] = time.monotonic() - 1

    result = await client.get_prefix("/prefix/")
    assert "/prefix/alive" in result
    assert "/prefix/dead" not in result


@pytest.mark.asyncio
async def test_reset_memory_clears_ttls():
    """reset_memory should clear TTLs as well."""
    client = EtcdClient(["memory://local"])
    await client.put("/key", "val", ttl=10)
    assert len(_GLOBAL_MEMORY_STORE.ttls) > 0

    EtcdClient.reset_memory()
    assert len(_GLOBAL_MEMORY_STORE.ttls) == 0
    assert len(_GLOBAL_MEMORY_STORE.values) == 0
    assert len(_GLOBAL_MEMORY_STORE.leader) == 0
