"""Reader FastAPI service for Phase 5."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from vecscaledb.config import Settings
from vecscaledb.coordinator.client import CoordinatorClient
from vecscaledb.models import NodeInfo, ReaderSearchRequest, SearchResult
from vecscaledb.observability import attach_observability, configure_logging
from vecscaledb.reader.search import SegmentCache


settings = Settings()
logger = configure_logging(settings)
coordinator_client: CoordinatorClient | None = None
cache: SegmentCache | None = None
refresh_task: asyncio.Task | None = None
first_refresh_complete = False


def create_coordinator_client() -> CoordinatorClient:
    return CoordinatorClient(settings.coordinator_urls)


def create_cache() -> SegmentCache:
    return SegmentCache(
        shared_storage=settings.shared_storage_path,
        shard_id=settings.reader_shard_id,
        max_segments_in_memory=settings.reader_max_segments_in_memory,
        load_all_shards=settings.reader_load_all_shards,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global coordinator_client, cache, refresh_task, first_refresh_complete
    first_refresh_complete = False
    coordinator_client = create_coordinator_client()
    cache = create_cache()
    await _register_self()
    await _refresh_once()
    refresh_task = asyncio.create_task(_refresh_loop())
    logger.info("reader.started", **_health_payload())
    yield
    if refresh_task is not None:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
    logger.info("reader.stopped")


app = FastAPI(title="VecScaleDB Reader", lifespan=lifespan)
attach_observability(app, settings, logger, lambda: _metrics_gauges())


@app.post("/search")
async def search(req: ReaderSearchRequest) -> SearchResult:
    active_cache = _require_cache()
    query = np.ascontiguousarray(np.array(req.query, dtype=np.float32))
    if query.shape != (settings.dim,):
        raise HTTPException(
            status_code=400,
            detail=f"Expected query with shape [{settings.dim}].",
        )
    ids, distances = active_cache.search_all(
        query=query,
        top_k=req.top_k,
        nprobe=req.nprobe,
        snapshot_id=req.snapshot_id,
    )
    return SearchResult(ids=ids, distances=distances)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **_health_payload()}


@app.get("/ready")
async def ready() -> dict:
    if not first_refresh_complete:
        raise HTTPException(status_code=503, detail="Initial segment refresh is not complete.")
    return {"status": "ready", **_health_payload()}


async def _refresh_loop() -> None:
    while True:
        try:
            await _register_self()
            await _refresh_once()
        except Exception as exc:
            logger.warning("reader.refresh.failed", error=str(exc))
        await asyncio.sleep(5)


async def _register_self() -> None:
    client = _require_coordinator_client()
    await client.register_node(
        NodeInfo(
            node_id=settings.node_id,
            role="reader",
            address=f"http://{settings.node_id}:8200",
            shard_ids=[settings.reader_shard_id],
        )
    )


async def _refresh_once() -> None:
    global first_refresh_complete
    client = _require_coordinator_client()
    active_cache = _require_cache()
    segments = await client.get_all_segments()
    await active_cache.refresh(segments)
    first_refresh_complete = True


def _health_payload() -> dict:
    active_cache = _require_cache()
    return {
        "shard_id": settings.reader_shard_id,
        "load_all_shards": settings.reader_load_all_shards,
        "segments_loaded": active_cache.segments_loaded,
        "ntotal": active_cache.ntotal,
        "segment_ids": active_cache.segment_ids,
    }


def _require_coordinator_client() -> CoordinatorClient:
    if coordinator_client is None:
        raise HTTPException(status_code=503, detail="Coordinator client is not ready.")
    return coordinator_client


def _require_cache() -> SegmentCache:
    if cache is None:
        raise HTTPException(status_code=503, detail="Segment cache is not ready.")
    return cache


def _metrics_gauges() -> dict:
    if cache is None:
        return {"segments_loaded": 0}
    return {
        "segments_loaded": cache.segments_loaded,
        "vectors_loaded": cache.ntotal,
        "reader_ready": first_refresh_complete,
    }


if __name__ == "__main__":
    uvicorn.run(
        "vecscaledb.reader.app:app",
        host="0.0.0.0",
        port=8200,
        log_level="info",
    )
