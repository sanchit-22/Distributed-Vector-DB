"""Writer ingest pipeline for Phase 4."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from vecscaledb.config import Settings
from vecscaledb.coordinator.client import CoordinatorClient
from vecscaledb.models import NodeInfo, SegmentMeta
from vecscaledb.storage.lsm import LSMEngine


class CoordinatorApi(Protocol):
    async def register_node(self, info: NodeInfo) -> None: ...

    async def register_segment(self, meta: SegmentMeta) -> None: ...

    async def delete_segment(self, segment_id: str) -> None: ...

    async def get_all_segments(self) -> list[SegmentMeta]: ...


class WriterPipeline:
    """Receives writes, persists them through LSMEngine, and registers metadata."""

    def __init__(
        self,
        config: Settings,
        coordinator_client: CoordinatorApi | None = None,
    ) -> None:
        self.config = config
        self.coordinator_client = coordinator_client or CoordinatorClient(
            config.coordinator_urls
        )
        self.lsm_engine = LSMEngine(
            shared_storage=config.shared_storage_path,
            wal_path=config.wal_path,
            shard_id=0,
            dim=config.dim,
            nlist=config.nlist,
            flush_threshold=config.segment_flush_threshold,
            merge_threshold=config.segment_merge_threshold,
        )
        self._registered_segments: set[str] = set()
        self._started = False

    async def start(self) -> None:
        await self.register_self()
        await self.lsm_engine.start()
        await self._load_registered_segments()
        await self.register_unregistered_segments()
        self._started = True

    async def stop(self) -> None:
        if self._started:
            await self.lsm_engine.stop()
            self._started = False

    async def insert(self, ids: np.ndarray, vectors: np.ndarray) -> dict:
        self._validate_vectors(ids, vectors)
        result = await self.lsm_engine.insert(ids, vectors)
        await self.register_unregistered_segments()
        health = await self.lsm_engine.health()
        return {
            "lsn": result["lsn"],
            "inserted": len(ids),
            "segment_flushed": result["segment_flushed"],
            "ntotal": health["ntotal"],
            "segments": health["segments"],
            "wal_lsn": health["wal_lsn"],
        }

    async def delete(self, ids: np.ndarray) -> dict:
        ids = np.ascontiguousarray(ids.astype(np.int64))
        lsn = await self.lsm_engine.delete(ids)
        health = await self.lsm_engine.health()
        return {
            "deleted": len(ids),
            "lsn": lsn,
            "ntotal": health["ntotal"],
            "wal_lsn": health["wal_lsn"],
        }

    async def health(self) -> dict:
        return await self.lsm_engine.health()

    async def register_self(self) -> None:
        await self.coordinator_client.register_node(self._node_info())

    async def register_unregistered_segments(self) -> None:
        local_ids = {segment.segment_id for segment in self.lsm_engine.segments}
        remote_segments = await self.coordinator_client.get_all_segments()
        for segment in remote_segments:
            if segment.shard_id == 0 and segment.segment_id not in local_ids:
                await self.coordinator_client.delete_segment(segment.segment_id)
                self._registered_segments.discard(segment.segment_id)

        for segment in self.lsm_engine.segments:
            if segment.segment_id in self._registered_segments:
                continue
            await self.coordinator_client.register_segment(segment.meta)
            self._registered_segments.add(segment.segment_id)

    async def _load_registered_segments(self) -> None:
        try:
            registered = await self.coordinator_client.get_all_segments()
        except Exception:
            registered = []
        self._registered_segments = {segment.segment_id for segment in registered}

    def _node_info(self) -> NodeInfo:
        return NodeInfo(
            node_id=self.config.node_id,
            role="writer",
            address=self.config.writer_url,
            shard_ids=[],
        )

    def _validate_vectors(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.ascontiguousarray(ids.astype(np.int64))
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.config.dim:
            raise ValueError(f"Expected vectors with shape [N, {self.config.dim}].")
        if len(ids) != len(vectors):
            raise ValueError("len(ids) != len(vectors)")


def writer_address_from_env(config: Settings) -> str:
    return os.environ.get("VECSCALE_WRITER_URL", config.writer_url)
