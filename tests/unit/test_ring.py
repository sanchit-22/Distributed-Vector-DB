"""Tests for coordinator consistent hashing."""

import random
from collections import Counter

from vecscaledb.coordinator.ring import ConsistentHashRing


def test_ring_distribution_with_five_nodes():
    ring = ConsistentHashRing(virtual_nodes=150)
    for index in range(5):
        ring.add_node(f"reader-{index}")

    rng = random.Random(42)
    counts = Counter(ring.get_node(rng.randrange(10_000_000)) for _ in range(100_000))
    expected = 100_000 / 5

    for count in counts.values():
        assert abs(count - expected) / expected < 0.25


def test_ring_rebalance_remove_and_add_restores_owner():
    ring = ConsistentHashRing(virtual_nodes=100)
    for index in range(5):
        ring.add_node(f"reader-{index}")

    keys = list(range(0, 10_000, 17))
    before = {key: ring.get_node(key) for key in keys}

    ring.remove_node("reader-2")
    after_remove = {key: ring.get_node(key) for key in keys}
    assert all(owner != "reader-2" for owner in after_remove.values())

    ring.add_node("reader-2")
    after_add = {key: ring.get_node(key) for key in keys}
    assert after_add == before


def test_range_and_shard_helpers():
    ring = ConsistentHashRing(virtual_nodes=10)
    ring.add_node("reader-0")
    ring.add_node("reader-1")

    owners = ring.get_nodes_for_range(1, 100)

    assert owners <= {"reader-0", "reader-1"}
    assert ring.get_shard_id("reader-0") in {0, 1}
