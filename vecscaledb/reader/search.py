"""Reader-side segment cache and search execution."""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np

from vecscaledb.index.segment import Segment
from vecscaledb.models import SegmentMeta
from vecscaledb.storage.lsm import filter_result_ids, merge_results


class SegmentCache:
    """LRU cache of immutable segments owned by a reader shard."""

    def __init__(
        self,
        shared_storage: str,
        shard_id: int,
        max_segments_in_memory: int = 20,
        load_all_shards: bool = False,
    ) -> None:
        if max_segments_in_memory <= 0:
            raise ValueError("max_segments_in_memory must be positive.")
        self.shared_storage = shared_storage
        self.shard_id = shard_id
        self.max_segments_in_memory = max_segments_in_memory
        self.load_all_shards = load_all_shards
        self._segments: OrderedDict[str, Segment] = OrderedDict()
        self._tombstone_path = os.path.join(shared_storage, "tombstones.json")
        self._tombstones: set[int] = set()
        self._segment_hidden_ids: dict[str, set[int]] = {}
        self._visibility_signature: tuple[tuple[str, int], ...] = ()

    async def refresh(self, all_segments: list[SegmentMeta]) -> None:
        self._tombstones = self._load_tombstones()
        visible = {
            meta.segment_id: meta
            for meta in all_segments
            if self.load_all_shards or meta.shard_id == self.shard_id
        }

        for segment_id in list(self._segments.keys()):
            if segment_id not in visible:
                self._segments.pop(segment_id, None)

        for segment_id, meta in sorted(
            visible.items(), key=lambda item: (item[1].snapshot_id, item[0])
        ):
            if segment_id in self._segments:
                self._segments.move_to_end(segment_id)
                continue
            segment = Segment.load(meta.path)
            self._segments[segment_id] = segment
            self._segments.move_to_end(segment_id)
            while len(self._segments) > self.max_segments_in_memory:
                self._segments.popitem(last=False)
        self._refresh_visibility_index_if_needed()

    def search_all(
        self,
        query: np.ndarray,
        top_k: int,
        nprobe: int,
        snapshot_id: int,
    ) -> tuple[list[int], list[float]]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        results = []
        visible_segments = [
            (segment_id, segment)
            for segment_id, segment in list(self._segments.items())
            if segment.snapshot_id <= snapshot_id
        ]

        for segment_id, segment in visible_segments:
            distances, ids = segment.search(query, top_k, nprobe)
            hidden_ids = self._tombstones | self._segment_hidden_ids.get(segment_id, set())
            distances, ids = filter_result_ids(distances, ids, hidden_ids)
            results.append((distances, ids))
            self._segments.move_to_end(segment_id)
        return merge_results(results, top_k, self._tombstones)

    @property
    def segments_loaded(self) -> int:
        return len(self._segments)

    @property
    def ntotal(self) -> int:
        visible_ids: set[int] = set()
        for segment in self._segments.values():
            visible_ids.update(int(vector_id) for vector_id in segment.load_ids().tolist())
        return len(visible_ids - self._tombstones)

    @property
    def segment_ids(self) -> list[str]:
        return list(self._segments.keys())

    def _refresh_visibility_index_if_needed(self) -> None:
        signature = tuple(
            sorted(
                (segment.segment_id, segment.snapshot_id)
                for segment in self._segments.values()
            )
        )
        if signature == self._visibility_signature:
            return
        self._visibility_signature = signature
        self._segment_hidden_ids = self._build_segment_hidden_ids()

    def _build_segment_hidden_ids(self) -> dict[str, set[int]]:
        latest_snapshot_by_id: dict[int, int] = {}
        segment_ids: dict[str, np.ndarray] = {}

        for segment_id, segment in self._segments.items():
            ids = segment.load_ids()
            segment_ids[segment_id] = ids
            for vector_id in ids.tolist():
                vector_id = int(vector_id)
                latest_snapshot_by_id[vector_id] = max(
                    latest_snapshot_by_id.get(vector_id, -1),
                    segment.snapshot_id,
                )

        hidden_by_segment: dict[str, set[int]] = {}
        for segment_id, segment in self._segments.items():
            hidden_by_segment[segment_id] = {
                int(vector_id)
                for vector_id in segment_ids[segment_id].tolist()
                if latest_snapshot_by_id[int(vector_id)] > segment.snapshot_id
            }
        return hidden_by_segment

    def _load_tombstones(self) -> set[int]:
        if not os.path.isfile(self._tombstone_path):
            return set()
        try:
            with open(self._tombstone_path) as f:
                return set(int(vector_id) for vector_id in json.load(f))
        except (json.JSONDecodeError, TypeError, ValueError):
            return set()
