"""LSM-inspired storage engine for the single-node runtime."""

from __future__ import annotations

import asyncio
import heapq
import json
import os
import shutil
from dataclasses import dataclass

import faiss
import numpy as np

from vecscaledb.index.segment import Segment
from vecscaledb.storage.snapshot import SnapshotManager
from vecscaledb.storage.wal import OP_DELETE, OP_INSERT, WAL


@dataclass
class MemTableEntry:
    vector: np.ndarray | None
    lsn: int
    deleted: bool = False


def merge_results(
    results: list[tuple[np.ndarray, np.ndarray]],
    top_k: int,
    deleted_ids: set[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Merge result arrays, deduplicate by ID, and keep smallest distance."""
    deleted_ids = deleted_ids or set()
    seen: dict[int, float] = {}
    for distances, ids in results:
        for distance, vector_id in zip(distances.tolist(), ids.tolist()):
            vector_id = int(vector_id)
            if vector_id in deleted_ids:
                continue
            if vector_id not in seen or float(distance) < seen[vector_id]:
                seen[vector_id] = float(distance)

    top = heapq.nsmallest(top_k, seen.items(), key=lambda item: item[1])
    if not top:
        return [], []
    ids, distances = zip(*top)
    return list(ids), list(distances)


def filter_result_ids(
    distances: np.ndarray,
    ids: np.ndarray,
    hidden_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove search hits whose IDs are hidden by deletes or newer versions."""
    if not hidden_ids or len(ids) == 0:
        return distances, ids
    mask = np.array([int(vector_id) not in hidden_ids for vector_id in ids], dtype=bool)
    return distances[mask], ids[mask]


class MemTable:
    """In-memory write buffer with per-entry LSN visibility."""

    def __init__(self, dim: int, flush_threshold: int) -> None:
        self.dim = dim
        self.flush_threshold = flush_threshold
        self._entries: dict[int, MemTableEntry] = {}

    def insert(self, ids: np.ndarray, vectors: np.ndarray, lsn: int) -> None:
        ids, vectors = self._prepare(ids, vectors)
        for vector_id, vector in zip(ids.tolist(), vectors):
            self._entries[int(vector_id)] = MemTableEntry(
                vector=np.ascontiguousarray(vector, dtype=np.float32),
                lsn=int(lsn),
                deleted=False,
            )

    def delete(self, ids: np.ndarray, lsn: int) -> None:
        for vector_id in ids.astype(np.int64).tolist():
            self._entries[int(vector_id)] = MemTableEntry(
                vector=None,
                lsn=int(lsn),
                deleted=True,
            )

    def search(
        self, query: np.ndarray, top_k: int, snapshot_id: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        ids, vectors = self.to_arrays(snapshot_id)
        if len(ids) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        index = faiss.IndexFlatL2(self.dim)
        index_map = faiss.IndexIDMap2(index)
        index_map.add_with_ids(vectors, ids)
        query_2d = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        k = min(top_k, index_map.ntotal)
        distances, result_ids = index_map.search(query_2d, k)
        result_ids = result_ids[0]
        distances = distances[0]
        mask = result_ids >= 0
        return distances[mask], result_ids[mask]

    def should_flush(self) -> bool:
        return self.ntotal >= self.flush_threshold

    def to_arrays(
        self, snapshot_id: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for vector_id, entry in self._entries.items():
            if snapshot_id is not None and entry.lsn > snapshot_id:
                continue
            if entry.deleted or entry.vector is None:
                continue
            ids.append(vector_id)
            vectors.append(entry.vector)

        if not ids:
            return (
                np.array([], dtype=np.int64),
                np.empty((0, self.dim), dtype=np.float32),
            )
        return (
            np.array(ids, dtype=np.int64),
            np.ascontiguousarray(np.vstack(vectors), dtype=np.float32),
        )

    def visible_deleted_ids(self, snapshot_id: int | None = None) -> set[int]:
        deleted: set[int] = set()
        for vector_id, entry in self._entries.items():
            if snapshot_id is not None and entry.lsn > snapshot_id:
                continue
            if entry.deleted:
                deleted.add(vector_id)
        return deleted

    def clear(self) -> None:
        self._entries.clear()

    @property
    def ntotal(self) -> int:
        return sum(1 for entry in self._entries.values() if not entry.deleted)

    @property
    def max_lsn(self) -> int:
        if not self._entries:
            return 0
        return max(entry.lsn for entry in self._entries.values())

    def _prepare(
        self, ids: np.ndarray, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.ascontiguousarray(ids.astype(np.int64))
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape [N, {self.dim}].")
        if len(ids) != len(vectors):
            raise ValueError("len(ids) != len(vectors)")
        return ids, vectors


class LSMEngine:
    """WAL-backed segment engine used by the Phase 2 single-node app."""

    def __init__(
        self,
        shared_storage: str,
        wal_path: str,
        shard_id: int,
        dim: int,
        nlist: int,
        flush_threshold: int,
        merge_threshold: int,
    ) -> None:
        self.shared_storage = shared_storage
        self.wal_path = wal_path
        self.shard_id = shard_id
        self.dim = dim
        self.nlist = nlist
        self.flush_threshold = flush_threshold
        self.merge_threshold = merge_threshold
        self.segments_path = os.path.join(shared_storage, "segments")
        self._tombstone_path = os.path.join(shared_storage, "tombstones.json")
        self.wal = WAL(wal_path, dim)
        self.memtable = MemTable(dim, flush_threshold)
        self.snapshots = SnapshotManager()
        self._segments: list[Segment] = []
        self._tombstones: set[int] = set()
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        async with self._lock:
            self.wal.open()
            self._segments = self._load_existing_segments()
            self._tombstones = self._load_tombstones()
            max_segment_lsn = max((seg.snapshot_id for seg in self._segments), default=0)
            self.memtable.clear()
            for record in self.wal.recover():
                if record.lsn <= max_segment_lsn:
                    continue
                if record.op_type == OP_INSERT:
                    self.memtable.insert(record.ids, record.vectors, record.lsn)
                elif record.op_type == OP_DELETE:
                    self.memtable.delete(record.ids, record.lsn)
                    self._tombstones.update(int(i) for i in record.ids.tolist())
            self._started = True

    async def stop(self) -> None:
        await self.flush()
        self.wal.close()
        self._started = False

    async def insert(self, ids: np.ndarray, vectors: np.ndarray) -> dict:
        self._ensure_started()
        ids = np.ascontiguousarray(ids.astype(np.int64))
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape [N, {self.dim}].")
        if len(ids) != len(vectors):
            raise ValueError("len(ids) != len(vectors)")

        async with self._lock:
            lsn = self.wal.append(OP_INSERT, ids, vectors)
            self.memtable.insert(ids, vectors, lsn)
            # Clear tombstones for re-inserted IDs
            inserted_ids = set(int(i) for i in ids.tolist())
            removed_tombstones = self._tombstones & inserted_ids
            if removed_tombstones:
                self._tombstones -= removed_tombstones
                self._persist_tombstones()
            should_flush = self.memtable.should_flush()

        flushed = False
        if should_flush:
            flushed = await self.flush() is not None
        return {"lsn": lsn, "segment_flushed": flushed}

    async def delete(self, ids: np.ndarray) -> int:
        self._ensure_started()
        ids = np.ascontiguousarray(ids.astype(np.int64))
        vectors = np.zeros((len(ids), self.dim), dtype=np.float32)
        async with self._lock:
            lsn = self.wal.append(OP_DELETE, ids, vectors)
            self.memtable.delete(ids, lsn)
            self._tombstones.update(int(i) for i in ids.tolist())
            self._persist_tombstones()
            return lsn

    async def flush(self) -> Segment | None:
        self._ensure_started()
        async with self._lock:
            # Capture memtable deletes into persistent tombstones before clearing
            mem_deletes = self.memtable.visible_deleted_ids()
            if mem_deletes:
                self._tombstones.update(mem_deletes)
                self._persist_tombstones()
            ids, vectors = self.memtable.to_arrays()
            snapshot_id = self.memtable.max_lsn
            if len(ids) == 0 or snapshot_id == 0:
                self.memtable.clear()
                return None
            self.memtable.clear()

        segment = Segment.create(
            self.shard_id,
            snapshot_id,
            vectors,
            ids,
            self.dim,
            self.nlist,
            self.segments_path,
        )

        async with self._lock:
            self._segments.append(segment)
            self._segments.sort(key=lambda seg: (seg.snapshot_id, seg.segment_id))
            self.wal.truncate_before(snapshot_id + 1)
            should_merge = len(self._segments) >= self.merge_threshold

        if should_merge:
            await self.merge()
        return segment

    async def merge(self) -> Segment | None:
        self._ensure_started()
        async with self._lock:
            if len(self._segments) < self.merge_threshold:
                return None
            to_merge = self._segments[: self.merge_threshold]
            remaining = self._segments[self.merge_threshold :]

        ids_list: list[np.ndarray] = []
        vectors_list: list[np.ndarray] = []
        latest: dict[int, tuple[int, np.ndarray]] = {}
        for segment in to_merge:
            ids, vectors = segment.load_arrays()
            for vector_id, vector in zip(ids.tolist(), vectors):
                latest[int(vector_id)] = (
                    segment.snapshot_id,
                    np.ascontiguousarray(vector, dtype=np.float32),
                )

        # Remove tombstoned IDs during merge (physical cleanup)
        async with self._lock:
            current_tombstones = set(self._tombstones)

        cleaned_tombstones: set[int] = set()
        for vector_id in sorted(latest):
            if int(vector_id) in current_tombstones:
                cleaned_tombstones.add(int(vector_id))
                continue
            ids_list.append(np.array([vector_id], dtype=np.int64))
            vectors_list.append(latest[vector_id][1].reshape(1, -1))

        if not ids_list:
            # All vectors were tombstoned — remove old segments, clear tombstones
            async with self._lock:
                self._segments = list(remaining)
                self._tombstones -= cleaned_tombstones
                self._persist_tombstones()
            for segment in to_merge:
                shutil.rmtree(segment.path, ignore_errors=True)
            return None

        merged_ids = np.concatenate(ids_list).astype(np.int64)
        merged_vectors = np.ascontiguousarray(np.vstack(vectors_list), dtype=np.float32)
        snapshot_id = max(seg.snapshot_id for seg in to_merge)

        merged = Segment.create(
            self.shard_id,
            snapshot_id,
            merged_vectors,
            merged_ids,
            self.dim,
            self.nlist,
            self.segments_path,
        )

        async with self._lock:
            self._segments = [merged, *remaining]
            self._segments.sort(key=lambda seg: (seg.snapshot_id, seg.segment_id))
            # Clear tombstones that were physically removed during merge
            self._tombstones -= cleaned_tombstones
            self._persist_tombstones()

        for segment in to_merge:
            shutil.rmtree(segment.path, ignore_errors=True)
        return merged

    async def search(
        self,
        query: np.ndarray,
        top_k: int,
        nprobe: int,
        snapshot_id: int | None = None,
    ) -> tuple[list[int], list[float]]:
        self._ensure_started()
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"Expected query with shape [{self.dim}].")

        created_snapshot = snapshot_id is None
        async with self._lock:
            if snapshot_id is None:
                snapshot_id = self.snapshots.create_snapshot(self.wal.last_lsn)
            visible_segments = self.snapshots.visible_segments(
                snapshot_id, list(self._segments)
            )
            visible_mem_ids = set(self.memtable.to_arrays(snapshot_id)[0].tolist())
            memtable_result = self.memtable.search(query, top_k, snapshot_id)
            deleted_ids = self.memtable.visible_deleted_ids(snapshot_id) | self._tombstones
            latest_segment_snapshot = self._latest_segment_snapshot_by_id(
                visible_segments
            )

        try:
            results = []
            for segment in visible_segments:
                distances, ids = segment.search(query, top_k, nprobe)
                hidden_ids = deleted_ids | visible_mem_ids | {
                    vector_id
                    for vector_id, latest_snapshot in latest_segment_snapshot.items()
                    if latest_snapshot > segment.snapshot_id
                }
                distances, ids = filter_result_ids(distances, ids, hidden_ids)
                results.append((distances, ids))
            if len(memtable_result[0]) > 0:
                results.append(memtable_result)
            return merge_results(results, top_k, deleted_ids)
        finally:
            if created_snapshot:
                async with self._lock:
                    self.snapshots.release_snapshot(snapshot_id)

    async def health(self) -> dict:
        async with self._lock:
            return {
                "ntotal": self.ntotal,
                "segments": len(self._segments),
                "memtable_size": self.memtable.ntotal,
                "wal_lsn": self.wal.last_lsn,
                "active_snapshots": self.snapshots.active_count,
            }

    @property
    def segments(self) -> list[Segment]:
        return list(self._segments)

    @property
    def ntotal(self) -> int:
        segment_ids: set[int] = set()
        for segment in self._segments:
            ids = np.load(os.path.join(segment.path, "ids.npy"))
            segment_ids.update(int(vector_id) for vector_id in ids.tolist())
        mem_ids = set(self.memtable.to_arrays()[0].tolist())
        deleted_ids = self.memtable.visible_deleted_ids() | self._tombstones
        return len((segment_ids | mem_ids) - deleted_ids)

    @staticmethod
    def _latest_segment_snapshot_by_id(segments: list[Segment]) -> dict[int, int]:
        latest: dict[int, int] = {}
        for segment in segments:
            for vector_id in segment.load_ids().tolist():
                latest[int(vector_id)] = max(
                    latest.get(int(vector_id), -1), segment.snapshot_id
                )
        return latest

    def _load_existing_segments(self) -> list[Segment]:
        if not os.path.isdir(self.segments_path):
            return []
        segments = []
        for entry in sorted(os.listdir(self.segments_path)):
            seg_dir = os.path.join(self.segments_path, entry)
            if os.path.isfile(os.path.join(seg_dir, "meta.json")):
                segment = Segment.load(seg_dir)
                if segment.shard_id == self.shard_id:
                    segments.append(segment)
        segments.sort(key=lambda seg: (seg.snapshot_id, seg.segment_id))
        return segments

    def _load_tombstones(self) -> set[int]:
        """Load the persisted tombstone set from shared storage."""
        if not os.path.isfile(self._tombstone_path):
            return set()
        try:
            with open(self._tombstone_path) as f:
                return set(int(vector_id) for vector_id in json.load(f))
        except (json.JSONDecodeError, TypeError, ValueError):
            return set()

    def _persist_tombstones(self) -> None:
        """Persist the current tombstone set to shared storage."""
        os.makedirs(os.path.dirname(self._tombstone_path), exist_ok=True)
        with open(self._tombstone_path, "w") as f:
            json.dump(sorted(self._tombstones), f)

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("LSMEngine is not started.")
