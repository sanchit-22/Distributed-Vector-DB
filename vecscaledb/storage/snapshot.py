"""Snapshot bookkeeping for stable point-in-time searches."""

from __future__ import annotations

from collections import Counter

from vecscaledb.index.segment import Segment


class SnapshotManager:
    """Tracks active read snapshots represented by WAL LSNs."""

    def __init__(self) -> None:
        self._active: Counter[int] = Counter()

    def create_snapshot(self, current_lsn: int) -> int:
        snapshot_id = int(current_lsn)
        self._active[snapshot_id] += 1
        return snapshot_id

    def release_snapshot(self, snapshot_id: int) -> None:
        if self._active[snapshot_id] <= 1:
            self._active.pop(snapshot_id, None)
        else:
            self._active[snapshot_id] -= 1

    def visible_segments(
        self, snapshot_id: int, all_segments: list[Segment]
    ) -> list[Segment]:
        return [seg for seg in all_segments if seg.snapshot_id <= snapshot_id]

    @property
    def active_count(self) -> int:
        return sum(self._active.values())
