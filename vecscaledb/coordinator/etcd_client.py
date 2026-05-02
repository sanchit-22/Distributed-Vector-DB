"""Coordinator metadata client with etcd and in-memory backends."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class _MemoryStore:
    values: dict[str, str]
    leader: dict[str, str]


_GLOBAL_MEMORY_STORE = _MemoryStore(values={}, leader={})


class EtcdClient:
    """Small async wrapper around etcd3 with retry support.

    Tests and local development can use the in-memory backend by passing
    ``["memory://local"]`` as endpoints.
    """

    def __init__(self, endpoints: list[str], backend: str | None = None) -> None:
        self.endpoints = endpoints
        self.backend = backend or (
            "memory" if any(endpoint.startswith("memory://") for endpoint in endpoints) else "etcd"
        )
        self._memory = _GLOBAL_MEMORY_STORE
        self._client: Any | None = None
        if self.backend == "etcd":
            self._client = self._connect_etcd(endpoints)

    async def put(self, key: str, value: str, ttl: int | None = None) -> None:
        if self.backend == "memory":
            self._memory.values[key] = value
            return

        def op() -> None:
            lease = self._client.lease(ttl) if ttl else None
            self._client.put(key, value, lease=lease)

        await self._retry(op)

    async def get(self, key: str) -> str | None:
        if self.backend == "memory":
            return self._memory.values.get(key)

        def op() -> str | None:
            value, _ = self._client.get(key)
            if value is None:
                return None
            return value.decode("utf-8")

        return await self._retry(op)

    async def get_prefix(self, prefix: str) -> dict[str, str]:
        if self.backend == "memory":
            return {
                key: value
                for key, value in sorted(self._memory.values.items())
                if key.startswith(prefix)
            }

        def op() -> dict[str, str]:
            items = {}
            for value, metadata in self._client.get_prefix(prefix):
                items[metadata.key.decode("utf-8")] = value.decode("utf-8")
            return items

        return await self._retry(op)

    async def delete(self, key: str) -> None:
        if self.backend == "memory":
            self._memory.values.pop(key, None)
            return
        await self._retry(lambda: self._client.delete(key))

    async def watch_prefix(self, prefix: str) -> AsyncIterator[tuple[str, str]]:
        seen: dict[str, str] = {}
        while True:
            current = await self.get_prefix(prefix)
            for key, value in current.items():
                if seen.get(key) != value:
                    seen[key] = value
                    yield key, value
            for key in set(seen) - set(current):
                seen.pop(key, None)
                yield key, ""
            await asyncio.sleep(1.0)

    async def campaign(self, election_name: str, value: str) -> bool:
        leader_key = f"/vecscaledb/elections/{election_name}"
        if self.backend == "memory":
            self._memory.leader.setdefault(election_name, value)
            self._memory.values["/vecscaledb/leader"] = self._memory.leader[election_name]
            return self._memory.leader[election_name] == value

        current = await self.get(leader_key)
        if current is None:
            await self.put(leader_key, value, ttl=10)
            await self.put("/vecscaledb/leader", value, ttl=10)
            return True
        return current == value

    async def observe_leader(self, election_name: str) -> AsyncIterator[str]:
        leader_key = f"/vecscaledb/elections/{election_name}"
        while True:
            leader = await self.get(leader_key)
            if leader is not None:
                yield leader
            await asyncio.sleep(1.0)

    @classmethod
    def reset_memory(cls) -> None:
        _GLOBAL_MEMORY_STORE.values.clear()
        _GLOBAL_MEMORY_STORE.leader.clear()

    @staticmethod
    def _connect_etcd(endpoints: list[str]) -> Any:
        try:
            import etcd3
        except ImportError as exc:
            raise RuntimeError(
                "etcd3 is required for the real coordinator backend. "
                "Install it or use memory://local for tests."
            ) from exc

        first = endpoints[0].replace("http://", "").replace("https://", "")
        host, _, port = first.partition(":")
        return etcd3.client(host=host, port=int(port or "2379"))

    async def _retry(self, fn: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await asyncio.to_thread(fn)
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.1 * (2**attempt))
        raise last_error  # type: ignore[misc]
