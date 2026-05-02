"""Consistent hashing ring used by the coordinator."""

from __future__ import annotations

import bisect
import hashlib


class ConsistentHashRing:
    """Deterministic consistent hash ring with virtual nodes."""

    def __init__(self, virtual_nodes: int = 150) -> None:
        if virtual_nodes <= 0:
            raise ValueError("virtual_nodes must be positive.")
        self.virtual_nodes = virtual_nodes
        self._ring: list[tuple[int, str]] = []
        self._nodes: set[str] = set()

    def add_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            return
        self._nodes.add(node_id)
        for replica in range(self.virtual_nodes):
            token = self._hash(f"{node_id}:{replica}")
            bisect.insort(self._ring, (token, node_id))

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        self._nodes.remove(node_id)
        self._ring = [(token, node) for token, node in self._ring if node != node_id]

    def get_node(self, key: int | str) -> str:
        if not self._ring:
            raise ValueError("Cannot route key on an empty ring.")
        token = self._hash(str(key))
        index = bisect.bisect_left(self._ring, (token, ""))
        if index == len(self._ring):
            index = 0
        return self._ring[index][1]

    def get_nodes_for_range(self, start_id: int, end_id: int) -> set[str]:
        if start_id > end_id:
            raise ValueError("start_id must be <= end_id.")
        if not self._ring:
            return set()
        return {self.get_node(key) for key in range(start_id, end_id + 1)}

    def get_shard_id(self, node_id: str) -> int:
        if not self._nodes:
            raise ValueError("Cannot compute shard id without nodes.")
        return self._hash(node_id) % len(self._nodes)

    def clear(self) -> None:
        self._ring.clear()
        self._nodes.clear()

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    @property
    def ring_size(self) -> int:
        return len(self._ring)

    @staticmethod
    def _hash(value: str) -> int:
        digest = hashlib.md5(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)
