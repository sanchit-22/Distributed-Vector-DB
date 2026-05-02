"""Writer FastAPI service for Phase 4."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from vecscaledb.config import Settings
from vecscaledb.models import DeleteRequest, InsertRequest
from vecscaledb.observability import attach_observability, configure_logging
from vecscaledb.writer.pipeline import WriterPipeline


settings = Settings()
logger = configure_logging(settings)
pipeline: WriterPipeline | None = None
segment_registration_task: asyncio.Task | None = None


def create_pipeline() -> WriterPipeline:
    return WriterPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, segment_registration_task
    pipeline = create_pipeline()
    await pipeline.start()
    segment_registration_task = asyncio.create_task(_segment_registration_loop())
    logger.info("writer.started", **await pipeline.health())
    yield
    if segment_registration_task is not None:
        segment_registration_task.cancel()
        try:
            await segment_registration_task
        except asyncio.CancelledError:
            pass
    if pipeline is not None:
        await pipeline.register_unregistered_segments()
        await pipeline.stop()
        logger.info("writer.stopped")


app = FastAPI(title="VecScaleDB Writer", lifespan=lifespan)
attach_observability(app, settings, logger, lambda: _metrics_gauges())


@app.post("/insert")
async def insert(req: InsertRequest) -> dict:
    active_pipeline = _require_pipeline()
    ids = np.ascontiguousarray(np.array(req.ids, dtype=np.int64))
    try:
        vectors = np.ascontiguousarray(np.array(req.vectors, dtype=np.float32))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Expected vectors with shape [N, {settings.dim}].",
        ) from exc
    try:
        return await active_pipeline.insert(ids, vectors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/delete")
async def delete(req: DeleteRequest) -> dict:
    active_pipeline = _require_pipeline()
    ids = np.ascontiguousarray(np.array(req.ids, dtype=np.int64))
    return await active_pipeline.delete(ids)


@app.get("/health")
async def health() -> dict:
    active_pipeline = _require_pipeline()
    return {"status": "ok", **await active_pipeline.health()}


async def _segment_registration_loop() -> None:
    while True:
        try:
            if pipeline is not None:
                await pipeline.register_self()
                await pipeline.register_unregistered_segments()
        except Exception as exc:
            logger.warning("writer.registration.failed", error=str(exc))
        await asyncio.sleep(5)


def _require_pipeline() -> WriterPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Writer pipeline is not ready.")
    return pipeline


async def _metrics_gauges() -> dict:
    if pipeline is None:
        return {"wal_lsn": 0, "segments_loaded": 0}
    health_data = await pipeline.health()
    return {
        "wal_lsn": health_data.get("wal_lsn", 0),
        "segments_loaded": health_data.get("segments", 0),
        "memtable_size": health_data.get("memtable_size", 0),
        "active_snapshots": health_data.get("active_snapshots", 0),
    }


if __name__ == "__main__":
    uvicorn.run(
        "vecscaledb.writer.app:app",
        host="0.0.0.0",
        port=8100,
        log_level="info",
    )
