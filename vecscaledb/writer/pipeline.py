"""Writer ingest pipeline for Phase 4 with multi-shard support."""

from __future__ import annotations

import os
from collections import defaultdict
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
    """Receives writes, persists them through LSMEngine(s), and registers metadata.

    When ``num_shards > 1`` the pipeline creates one :class:`LSMEngine` per
    shard and distributes incoming vectors by hashing their IDs.  Each shard
    produces segments tagged with its ``shard_id`` so that readers in
    scatter-gather mode can load only the segments they own.
    """

    def __init__(
        self,
        config: Settings,
        coordinator_client: CoordinatorApi | None = None,
    ) -> None:
        self.config = config
        self.coordinator_client = coordinator_client or CoordinatorClient(
            config.coordinator_urls
        )
        self.num_shards = max(1, config.num_shards)
        self._engines: dict[int, LSMEngine] = {}
        for shard_id in range(self.num_shards):
            self._engines[shard_id] = LSMEngine(
                shared_storage=config.shared_storage_path,
                wal_path=os.path.join(config.wal_path, f"shard_{shard_id}")
                if self.num_shards > 1
                else config.wal_path,
                shard_id=shard_id,
                dim=config.dim,
                nlist=config.nlist,
                flush_threshold=config.segment_flush_threshold,
                merge_threshold=config.segment_merge_threshold,
            )
        self._registered_segments: set[str] = set()
        self._started = False

    # Backwards-compatible single-engine accessor
    @property
    def lsm_engine(self) -> LSMEngine:
        """Return the shard-0 engine (backwards compatible)."""
        return self._engines[0]

    async def start(self) -> None:
        await self.register_self()
        for engine in self._engines.values():
            await engine.start()
        await self._load_registered_segments()
        await self.register_unregistered_segments()
        self._started = True

    async def stop(self) -> None:
        if self._started:
            for engine in self._engines.values():
                await engine.stop()
            self._started = False

    async def insert(self, ids: np.ndarray, vectors: np.ndarray) -> dict:
        self._validate_vectors(ids, vectors)

        if self.num_shards == 1:
            # Fast path: single shard
            result = await self._engines[0].insert(ids, vectors)
        else:
            # Distribute vectors across shards by hashing their IDs
            shard_groups: dict[int, tuple[list[int], list[np.ndarray]]] = defaultdict(
                lambda: ([], [])
            )
            for i, vec_id in enumerate(ids.tolist()):
                shard_id = int(vec_id) % self.num_shards
                id_list, vec_list = shard_groups[shard_id]
                id_list.append(int(vec_id))
                vec_list.append(vectors[i])

            result = {"lsn": 0, "segment_flushed": False}
            for shard_id, (id_list, vec_list) in shard_groups.items():
                shard_ids = np.array(id_list, dtype=np.int64)
                shard_vecs = np.ascontiguousarray(
                    np.vstack(vec_list), dtype=np.float32
                )
                shard_result = await self._engines[shard_id].insert(
                    shard_ids, shard_vecs
                )
                result["lsn"] = max(result["lsn"], shard_result["lsn"])
                result["segment_flushed"] = (
                    result["segment_flushed"] or shard_result["segment_flushed"]
                )

        await self.register_unregistered_segments()
        health = await self.health()
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

        if self.num_shards == 1:
            lsn = await self._engines[0].delete(ids)
        else:
            lsn = 0
            shard_groups: dict[int, list[int]] = defaultdict(list)
            for vec_id in ids.tolist():
                shard_groups[int(vec_id) % self.num_shards].append(int(vec_id))
            for shard_id, id_list in shard_groups.items():
                shard_lsn = await self._engines[shard_id].delete(
                    np.array(id_list, dtype=np.int64)
                )
                lsn = max(lsn, shard_lsn)

        health = await self.health()
        return {
            "deleted": len(ids),
            "lsn": lsn,
            "ntotal": health["ntotal"],
            "wal_lsn": health["wal_lsn"],
        }

    async def health(self) -> dict:
        combined = {"ntotal": 0, "segments": 0, "memtable_size": 0, "wal_lsn": 0, "active_snapshots": 0}
        for engine in self._engines.values():
            h = await engine.health()
            combined["ntotal"] += h.get("ntotal", 0)
            combined["segments"] += h.get("segments", 0)
            combined["memtable_size"] += h.get("memtable_size", 0)
            combined["wal_lsn"] = max(combined["wal_lsn"], h.get("wal_lsn", 0))
            combined["active_snapshots"] += h.get("active_snapshots", 0)
        return combined

    @property
    def segments(self) -> list:
        """Return all segments across all shard engines."""
        all_segments = []
        for engine in self._engines.values():
            all_segments.extend(engine.segments)
        return all_segments

    async def register_self(self) -> None:
        await self.coordinator_client.register_node(self._node_info())

    async def register_unregistered_segments(self) -> None:
        local_ids = {segment.segment_id for segment in self.segments}
        remote_segments = await self.coordinator_client.get_all_segments()

        # Clean up stale remote segment registrations
        local_shard_ids = set(self._engines.keys())
        for segment in remote_segments:
            if segment.shard_id in local_shard_ids and segment.segment_id not in local_ids:
                await self.coordinator_client.delete_segment(segment.segment_id)
                self._registered_segments.discard(segment.segment_id)

        # Register new local segments
        for segment in self.segments:
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
            shard_ids=list(self._engines.keys()),
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
