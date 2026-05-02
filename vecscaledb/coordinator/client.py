"""HTTP helper for talking to coordinator replicas."""

from __future__ import annotations

import asyncio

import httpx

from vecscaledb.models import NodeInfo, SegmentMeta


class CoordinatorClient:
    """Retrying client that fails over across configured coordinator URLs."""

    def __init__(self, coordinator_urls: list[str], timeout: float = 5.0) -> None:
        if not coordinator_urls:
            raise ValueError("At least one coordinator URL is required.")
        self.coordinator_urls = [url.rstrip("/") for url in coordinator_urls]
        self.timeout = timeout

    async def register_node(self, info: NodeInfo) -> None:
        await self._request("POST", "/register", json=info.model_dump())

    async def register_segment(self, meta: SegmentMeta) -> None:
        await self._request("POST", "/segments", json=meta.model_dump())

    async def delete_segment(self, segment_id: str) -> None:
        await self._request("DELETE", f"/segments/{segment_id}")

    async def get_all_segments(self) -> list[SegmentMeta]:
        data = await self._request("GET", "/segments")
        return [SegmentMeta.model_validate(item) for item in data]

    async def get_search_targets(self) -> list[NodeInfo]:
        data = await self._request("GET", "/route/search")
        return [NodeInfo.model_validate(item) for item in data.get("readers", [])]

    async def _request(self, method: str, path: str, **kwargs) -> object:
        last_error: Exception | None = None
        for attempt in range(3):
            for base_url in self.coordinator_urls:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.request(method, f"{base_url}{path}", **kwargs)
                        response.raise_for_status()
                        return response.json()
                except Exception as exc:
                    last_error = exc
            await asyncio.sleep(0.1 * (2**attempt))
        raise RuntimeError("All coordinator requests failed.") from last_error
