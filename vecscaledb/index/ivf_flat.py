"""IVF_FLAT index wrapper around Faiss."""

from __future__ import annotations

import sys

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
        self._index: faiss.Index | None = None
        self._quantizer: faiss.IndexFlatL2 | None = None
        self._ivf: faiss.IndexIVFFlat | None = None
        self._flat: faiss.IndexFlatL2 | None = None
        self._loaded_index: faiss.Index | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def train(self, vectors: np.ndarray) -> None:
        """Train the IVF quantizer on *vectors* (shape ``[N, dim]``, N >= nlist)."""
        vectors = self._prepare(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape [N, {self.dim}].")

        if sys.platform == "win32":
            # faiss-cpu 1.8.0 has a native access-violation failure in
            # IndexIVFFlat.search on Windows. Use exact flat search there so
            # development/tests remain stable; Docker/Linux still uses IVF.
            flat = faiss.IndexFlatL2(self.dim)
            self._index = faiss.IndexIDMap2(flat)
            self._flat = flat
            self._quantizer = None
            self._ivf = None
            self._trained = True
            return

        quantizer = faiss.IndexFlatL2(self.dim)
        ivf = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        ivf.train(vectors)
        self._index = ivf
        # Keep the quantizer wrapper alive with the IVF index. The Faiss C++
        # object stores a pointer to it, and Windows builds are unforgiving if
        # Python collects the wrapper too early.
        self._quantizer = quantizer
        self._ivf = ivf
        self._loaded_index = None
        self._trained = True

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add *vectors* with external *ids* (shape ``[N]``, uint64)."""
        if not self._trained:
            raise RuntimeError("Index must be trained before adding vectors.")
        if self._index is None:
            raise RuntimeError("Index is not initialized.")
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

        if self._ivf is not None:
            self._ivf.nprobe = nprobe

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
        loaded = faiss.read_index(path)
        downcasted = faiss.downcast_index(loaded)
        self._loaded_index = loaded
        self._index = loaded
        self._ivf = self._extract_ivf(downcasted)
        self._quantizer = None
        self._flat = None
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

    @staticmethod
    def _extract_ivf(index: faiss.Index) -> faiss.IndexIVFFlat | None:
        """Return an IVF index when the loaded Faiss index contains one."""
        if hasattr(index, "nprobe"):
            return index
        if hasattr(index, "index"):
            nested = faiss.downcast_index(index.index)
            if hasattr(nested, "nprobe"):
                return nested
        return None
