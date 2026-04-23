"""Single-node FastAPI server for Phase 1.

Provides ``/insert``, ``/search``, and ``/health`` endpoints.
Buffers vectors in a MemTable (dict) and flushes to immutable Segments.
This module is replaced by the distributed writer/reader apps in later phases.
"""

from __future__ import annotations

import asyncio
import heapq
import os
import uuid
from contextlib import asynccontextmanager

import faiss
import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI

from vecscaledb.config import Settings
from vecscaledb.index.segment import Segment
from vecscaledb.models import InsertRequest, SearchRequest, SearchResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
settings = Settings()

# MemTable: maps vector ID → np.ndarray (1-D, float32)
_memtable: dict[int, np.ndarray] = {}
_memtable_lock = asyncio.Lock()

# Flushed segments
_segments: list[Segment] = []
_segments_lock = asyncio.Lock()

_next_snapshot_id: int = 0
_snapshot_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memtable_search(query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force search over the current MemTable using ``IndexFlatL2``."""
    if not _memtable:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    ids = np.array(list(_memtable.keys()), dtype=np.int64)
    vecs = np.array(list(_memtable.values()), dtype=np.float32)
    vecs = np.ascontiguousarray(vecs)

    idx = faiss.IndexFlatL2(settings.dim)
    idx_map = faiss.IndexIDMap2(idx)
    idx_map.add_with_ids(vecs, ids)

    query_2d = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    k = min(top_k, idx_map.ntotal)
    if k == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    distances, result_ids = idx_map.search(query_2d, k)
    distances = distances[0]
    result_ids = result_ids[0]
    mask = result_ids >= 0
    return distances[mask], result_ids[mask]


def _merge_results(
    results: list[tuple[np.ndarray, np.ndarray]], top_k: int
) -> tuple[list[int], list[float]]:
    """Merge multiple ``(distances, ids)`` result lists, deduplicate, return top-k."""
    seen: dict[int, float] = {}
    for distances, ids in results:
        for d, i in zip(distances.tolist(), ids.tolist()):
            i_int = int(i)
            if i_int not in seen or d < seen[i_int]:
                seen[i_int] = d

    top = heapq.nsmallest(top_k, seen.items(), key=lambda x: x[1])
    if not top:
        return [], []
    result_ids, result_dists = zip(*top)
    return list(result_ids), list(result_dists)


async def _flush() -> None:
    """Flush the current MemTable to a new immutable Segment."""
    global _next_snapshot_id

    async with _memtable_lock:
        if not _memtable:
            return
        ids = np.array(list(_memtable.keys()), dtype=np.int64)
        vecs = np.array(list(_memtable.values()), dtype=np.float32)
        _memtable.clear()

    async with _snapshot_lock:
        snapshot_id = _next_snapshot_id
        _next_snapshot_id += 1

    base_path = os.path.join(settings.shared_storage_path, "segments")

    # Offload CPU-heavy segment creation to a thread
    loop = asyncio.get_event_loop()
    segment = await loop.run_in_executor(
        None,
        Segment.create,
        0,              # shard_id
        snapshot_id,
        vecs,
        ids,
        settings.dim,
        settings.nlist,
        base_path,
    )

    async with _segments_lock:
        _segments.append(segment)

    logger.info(
        "segment.flushed",
        segment_id=segment.segment_id,
        num_vectors=segment.ntotal,
        snapshot_id=snapshot_id,
    )


def _load_existing_segments() -> None:
    """Load any existing segments from shared storage on startup."""
    base_path = os.path.join(settings.shared_storage_path, "segments")
    if not os.path.isdir(base_path):
        return
    for entry in sorted(os.listdir(base_path)):
        seg_dir = os.path.join(base_path, entry)
        if os.path.isfile(os.path.join(seg_dir, "meta.json")):
            seg = Segment.load(seg_dir)
            _segments.append(seg)
            logger.info("segment.loaded", segment_id=seg.segment_id)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load existing segments on startup; flush memtable on shutdown."""
    _load_existing_segments()
    logger.info("node.started", segments=len(_segments))
    yield
    # Graceful shutdown: flush any remaining memtable data
    if _memtable:
        await _flush()
    logger.info("node.stopped")


app = FastAPI(title="VecScaleDB — Single Node", lifespan=lifespan)


@app.post("/insert")
async def insert(req: InsertRequest) -> dict:
    """Buffer vectors in the MemTable; flush to Segment when threshold is reached."""
    vectors = np.array(req.vectors, dtype=np.float32)
    ids = np.array(req.ids, dtype=np.int64)

    if vectors.shape[1] != settings.dim:
        return {"error": f"Expected dim={settings.dim}, got {vectors.shape[1]}"}
    if len(ids) != len(vectors):
        return {"error": "len(ids) != len(vectors)"}

    flushed = False
    async with _memtable_lock:
        for i, vid in enumerate(ids):
            _memtable[int(vid)] = vectors[i]

        if len(_memtable) >= settings.segment_flush_threshold:
            # Release lock, then flush
            pass

    if len(_memtable) >= settings.segment_flush_threshold:
        await _flush()
        flushed = True

    total = sum(s.ntotal for s in _segments) + len(_memtable)
    return {"inserted": len(ids), "segment_flushed": flushed, "ntotal": total}


@app.post("/search")
async def search(req: SearchRequest) -> SearchResult:
    """Search across all segments + memtable, merge results."""
    query = np.array(req.query, dtype=np.float32)
    results: list[tuple[np.ndarray, np.ndarray]] = []

    # Search flushed segments
    async with _segments_lock:
        segments_snapshot = list(_segments)

    loop = asyncio.get_event_loop()
    for seg in segments_snapshot:
        try:
            d, i = await loop.run_in_executor(
                None, seg.search, query, req.top_k, req.nprobe
            )
            results.append((d, i))
        except Exception:
            # Skip empty or broken segments
            pass

    # Search memtable
    async with _memtable_lock:
        d, i = _memtable_search(query, req.top_k)
    if len(d) > 0:
        results.append((d, i))

    merged_ids, merged_dists = _merge_results(results, req.top_k)
    return SearchResult(ids=merged_ids, distances=merged_dists)


@app.get("/health")
async def health() -> dict:
    total = sum(s.ntotal for s in _segments) + len(_memtable)
    return {
        "status": "ok",
        "ntotal": total,
        "segments": len(_segments),
        "memtable_size": len(_memtable),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "vecscaledb.node:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
