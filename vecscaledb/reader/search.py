"""Reader-side segment cache and search execution."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from vecscaledb.index.segment import Segment
from vecscaledb.models import SegmentMeta
from vecscaledb.storage.lsm import merge_results


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

    async def refresh(self, all_segments: list[SegmentMeta]) -> None:
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

    def search_all(
        self,
        query: np.ndarray,
        top_k: int,
        nprobe: int,
        snapshot_id: int,
    ) -> tuple[list[int], list[float]]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        results = []
        for segment_id, segment in list(self._segments.items()):
            if segment.snapshot_id > snapshot_id:
                continue
            distances, ids = segment.search(query, top_k, nprobe)
            results.append((distances, ids))
            self._segments.move_to_end(segment_id)
        return merge_results(results, top_k)

    @property
    def segments_loaded(self) -> int:
        return len(self._segments)

    @property
    def ntotal(self) -> int:
        return sum(segment.ntotal for segment in self._segments.values())

    @property
    def segment_ids(self) -> list[str]:
        return list(self._segments.keys())
