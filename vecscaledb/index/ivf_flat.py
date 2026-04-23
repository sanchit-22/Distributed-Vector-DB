"""IVF_FLAT index wrapper around Faiss."""

from __future__ import annotations

import faiss
import numpy as np


class IVFFlatIndex:
    """IVF_FLAT vector index backed by Faiss.

    Supports train → add → search lifecycle with external uint64 IDs.
    """

    def __init__(self, dim: int, nlist: int) -> None:
        self.dim = dim
        self.nlist = nlist
        self._trained: bool = False
        self._index: faiss.IndexIDMap2 | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def train(self, vectors: np.ndarray) -> None:
        """Train the IVF quantizer on *vectors* (shape ``[N, dim]``, N >= nlist)."""
        vectors = self._prepare(vectors)
        quantizer = faiss.IndexFlatL2(self.dim)
        ivf = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        ivf.train(vectors)
        self._index = faiss.IndexIDMap2(ivf)
        self._trained = True

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add *vectors* with external *ids* (shape ``[N]``, uint64)."""
        if not self._trained:
            raise RuntimeError("Index must be trained before adding vectors.")
        vectors = self._prepare(vectors)
        ids = np.ascontiguousarray(ids.astype(np.int64))
        self._index.add_with_ids(vectors, ids)

    def search(
        self, query: np.ndarray, top_k: int, nprobe: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for *top_k* nearest neighbours of *query* (shape ``[dim]``).

        Returns ``(distances, ids)`` each of shape ``[top_k]``.
        Unfilled slots (Faiss returns -1) are filtered out.
        """
        if self._index is None:
            raise RuntimeError("Index is not initialized.")

        # Set nprobe on the underlying IVF index
        ivf = faiss.downcast_index(self._index.index)
        ivf.nprobe = nprobe

        query = self._prepare(query.reshape(1, -1))
        distances, ids = self._index.search(query, top_k)

        # Flatten from [1, top_k] to [top_k]
        distances = distances[0]
        ids = ids[0]

        # Filter out unfilled slots (-1 ids)
        mask = ids >= 0
        return distances[mask], ids[mask]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to *path*."""
        if self._index is None:
            raise RuntimeError("No index to save.")
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        """Deserialize an index from *path*."""
        self._index = faiss.read_index(path)
        self._trained = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(arr: np.ndarray) -> np.ndarray:
        """Ensure float32 and C-contiguous layout (Faiss requirement)."""
        return np.ascontiguousarray(arr, dtype=np.float32)
