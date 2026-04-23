"""Immutable Segment abstraction — a persisted unit of indexed vectors."""

from __future__ import annotations

import json
import os
import uuid

import numpy as np

from vecscaledb.index.ivf_flat import IVFFlatIndex
from vecscaledb.models import SegmentMeta


class Segment:
    """An immutable, persisted collection of vectors with a Faiss IVF_FLAT index.

    A segment is stored on disk as a directory containing:
        - ``index.faiss``  — the serialized Faiss index
        - ``ids.npy``      — numpy array of external vector IDs
        - ``meta.json``    — serialized :class:`SegmentMeta`
    """

    def __init__(
        self,
        segment_id: str,
        shard_id: int,
        snapshot_id: int,
        path: str,
        meta: SegmentMeta,
        index: IVFFlatIndex,
    ) -> None:
        self.segment_id = segment_id
        self.shard_id = shard_id
        self.snapshot_id = snapshot_id
        self.path = path
        self.meta = meta
        self._index = index

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        shard_id: int,
        snapshot_id: int,
        vectors: np.ndarray,
        ids: np.ndarray,
        dim: int,
        nlist: int,
        base_path: str = "./shared_storage/segments",
    ) -> Segment:
        """Build and persist a new segment from raw vectors and IDs.

        Args:
            shard_id: logical shard this segment belongs to.
            snapshot_id: monotonically increasing version counter.
            vectors: ``[N, dim]`` float32.
            ids: ``[N]`` uint64 external vector IDs.
            dim: vector dimensionality.
            nlist: IVF_FLAT centroid count.
            base_path: root directory under which segment dirs are created.
        """
        segment_id = f"{shard_id:04d}-{snapshot_id:010d}-{uuid.uuid4().hex[:8]}"
        seg_dir = os.path.join(base_path, segment_id)
        os.makedirs(seg_dir, exist_ok=True)

        # Build index
        # Ensure nlist does not exceed the number of vectors
        effective_nlist = min(nlist, len(vectors))
        idx = IVFFlatIndex(dim, effective_nlist)
        idx.train(vectors)
        idx.add(vectors, ids)

        # Persist
        idx.save(os.path.join(seg_dir, "index.faiss"))
        np.save(os.path.join(seg_dir, "ids.npy"), ids)

        meta = SegmentMeta(
            segment_id=segment_id,
            shard_id=shard_id,
            num_vectors=int(len(ids)),
            snapshot_id=snapshot_id,
            path=seg_dir,
        )
        with open(os.path.join(seg_dir, "meta.json"), "w") as f:
            f.write(meta.model_dump_json(indent=2))

        return cls(
            segment_id=segment_id,
            shard_id=shard_id,
            snapshot_id=snapshot_id,
            path=seg_dir,
            meta=meta,
            index=idx,
        )

    @classmethod
    def load(cls, path: str) -> Segment:
        """Load a segment from an existing directory on disk."""
        with open(os.path.join(path, "meta.json")) as f:
            meta = SegmentMeta.model_validate_json(f.read())

        idx = IVFFlatIndex(dim=0, nlist=0)  # dim/nlist recovered from file
        idx.load(os.path.join(path, "index.faiss"))

        return cls(
            segment_id=meta.segment_id,
            shard_id=meta.shard_id,
            snapshot_id=meta.snapshot_id,
            path=path,
            meta=meta,
            index=idx,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, top_k: int, nprobe: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search this segment, returning ``(distances, ids)``."""
        return self._index.search(query, top_k, nprobe)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
