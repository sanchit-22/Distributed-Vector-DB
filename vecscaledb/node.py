"""Single-node FastAPI server backed by the Phase 2 LSM engine."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from vecscaledb.config import Settings
from vecscaledb.models import InsertRequest, SearchRequest, SearchResult
from vecscaledb.observability import attach_observability, configure_logging
from vecscaledb.storage.lsm import LSMEngine


settings = Settings()
logger = configure_logging(settings)
engine: LSMEngine | None = None


def create_engine() -> LSMEngine:
    return LSMEngine(
        shared_storage=settings.shared_storage_path,
        wal_path=settings.wal_path,
        shard_id=0,
        dim=settings.dim,
        nlist=settings.nlist,
        flush_threshold=settings.segment_flush_threshold,
        merge_threshold=settings.segment_merge_threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the WAL-backed engine and flush cleanly on shutdown."""
    global engine
    engine = create_engine()
    await engine.start()
    logger.info("node.started", **await engine.health())
    yield
    if engine is not None:
        await engine.stop()
        logger.info("node.stopped")


app = FastAPI(title="VecScaleDB - Single Node", lifespan=lifespan)
attach_observability(app, settings, logger, lambda: _metrics_gauges())


@app.post("/insert")
async def insert(req: InsertRequest) -> dict:
    """Insert vectors after appending them to the WAL."""
    active_engine = _require_engine()
    try:
        vectors = np.ascontiguousarray(np.array(req.vectors, dtype=np.float32))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Expected vectors with shape [N, {settings.dim}].",
        ) from exc
    ids = np.ascontiguousarray(np.array(req.ids, dtype=np.int64))

    if vectors.ndim != 2 or vectors.shape[1] != settings.dim:
        got = vectors.shape[1] if vectors.ndim == 2 else "invalid"
        raise HTTPException(
            status_code=400,
            detail=f"Expected vectors with shape [N, {settings.dim}], got {got}.",
        )
    if len(ids) != len(vectors):
        raise HTTPException(status_code=400, detail="len(ids) != len(vectors)")

    result = await active_engine.insert(ids, vectors)
    health = await active_engine.health()
    return {
        "inserted": len(ids),
        "segment_flushed": result["segment_flushed"],
        "ntotal": health["ntotal"],
        "lsn": result["lsn"],
    }


@app.post("/search")
async def search(req: SearchRequest) -> SearchResult:
    """Search a stable snapshot across visible segments and MemTable rows."""
    active_engine = _require_engine()
    query = np.ascontiguousarray(np.array(req.query, dtype=np.float32))
    if query.ndim != 1 or query.shape[0] != settings.dim:
        got = query.shape[0] if query.ndim == 1 else "invalid"
        raise HTTPException(
            status_code=400,
            detail=f"Expected query with {settings.dim} numbers, got {got}.",
        )
    ids, distances = await active_engine.search(
        query=query,
        top_k=req.top_k,
        nprobe=req.nprobe,
    )
    return SearchResult(ids=ids, distances=distances)


@app.get("/health")
async def health() -> dict:
    active_engine = _require_engine()
    engine_health = await active_engine.health()
    return {"status": "ok", **engine_health}


def _require_engine() -> LSMEngine:
    if engine is None:
        raise RuntimeError("LSM engine is not started.")
    return engine


async def _metrics_gauges() -> dict:
    if engine is None:
        return {"wal_lsn": 0, "segments_loaded": 0}
    health_data = await engine.health()
    return {
        "wal_lsn": health_data.get("wal_lsn", 0),
        "segments_loaded": health_data.get("segments", 0),
        "memtable_size": health_data.get("memtable_size", 0),
        "active_snapshots": health_data.get("active_snapshots", 0),
    }


if __name__ == "__main__":
    uvicorn.run(
        "vecscaledb.node:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
