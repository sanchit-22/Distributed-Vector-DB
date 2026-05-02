"""Coordinator FastAPI app for Phase 3 metadata routing."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query

from vecscaledb.config import Settings
from vecscaledb.coordinator.etcd_client import EtcdClient
from vecscaledb.coordinator.ring import ConsistentHashRing
from vecscaledb.models import NodeInfo, SearchRequest, SegmentMeta
from vecscaledb.observability import attach_observability, configure_logging
from vecscaledb.storage.lsm import merge_results


NODE_PREFIX = "/vecscaledb/nodes/"
SEGMENT_PREFIX = "/vecscaledb/segments/"
SNAPSHOT_CURRENT = "/vecscaledb/snapshots/current"


settings = Settings()
logger = configure_logging(settings)
etcd: EtcdClient | None = None
ring = ConsistentHashRing()
nodes: dict[str, NodeInfo] = {}
leader = False
heartbeat_task: asyncio.Task | None = None
search_counter = 0


def create_etcd_client() -> EtcdClient:
    return EtcdClient(settings.etcd_endpoints)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global etcd, leader, heartbeat_task
    etcd = create_etcd_client()
    leader = await etcd.campaign("coordinator", settings.node_id)
    await _register_self()
    await _refresh_nodes()
    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    logger.info("coordinator.started", leader=leader, nodes=len(nodes))
    yield
    if heartbeat_task is not None:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    logger.info("coordinator.stopped")


app = FastAPI(title="VecScaleDB Coordinator", lifespan=lifespan)
attach_observability(app, settings, logger, lambda: _metrics_gauges())


@app.get("/route/insert")
async def route_insert(count: int = Query(..., ge=1)) -> dict:
    writer = _find_writer()
    shard_owner = None
    shard_id = 0
    readers = _reader_nodes()
    if readers:
        shard_owner = ring.get_node(str(count))
        shard_id = ring.get_shard_id(shard_owner)
    return {
        "writer_url": writer.address if writer else settings.writer_url,
        "writer_node_id": writer.node_id if writer else None,
        "shard_id": shard_id,
        "shard_owner": shard_owner,
    }


@app.get("/route/search")
async def route_search(query_hash: int | None = None) -> dict:
    readers = _reader_nodes()
    if query_hash is not None and readers and ring.nodes:
        owner = ring.get_node(query_hash)
        readers = [node for node in readers if node.node_id == owner]
    return {"readers": [node.model_dump() for node in readers]}


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    client = _require_etcd()
    await _refresh_nodes()
    snapshot_id = await _current_snapshot_id(client)
    readers = _reader_nodes()
    if not readers:
        return {"ids": [], "distances": [], "incomplete": False}
    readers = _search_targets(readers)

    async with httpx.AsyncClient(timeout=5.0) as http_client:
        tasks = [
            _search_reader(http_client, reader, req, snapshot_id)
            for reader in readers
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    incomplete = False
    results = []
    for response in responses:
        if isinstance(response, Exception):
            incomplete = True
            logger.warning("reader.search.failed", error=str(response))
            continue
        distances = np.array(response.get("distances", []), dtype=np.float32)
        ids = np.array(response.get("ids", []), dtype=np.int64)
        results.append((distances, ids))

    ids, distances = merge_results(results, req.top_k)
    return {"ids": ids, "distances": distances, "incomplete": incomplete}


@app.post("/register")
async def register(info: NodeInfo) -> dict:
    if not leader:
        return await _proxy_to_leader("POST", "/register", info.model_dump())
    client = _require_etcd()
    await client.put(f"{NODE_PREFIX}{info.node_id}", info.model_dump_json(), ttl=10)
    nodes[info.node_id] = info
    _rebuild_ring()
    return {"registered": True, "node_id": info.node_id}


@app.get("/segments")
async def list_segments() -> list[SegmentMeta]:
    client = _require_etcd()
    values = await client.get_prefix(SEGMENT_PREFIX)
    return [SegmentMeta.model_validate_json(value) for value in values.values()]


@app.post("/segments")
async def register_segment(meta: SegmentMeta) -> dict:
    if not leader:
        return await _proxy_to_leader("POST", "/segments", meta.model_dump())
    client = _require_etcd()
    await client.put(f"{SEGMENT_PREFIX}{meta.segment_id}", meta.model_dump_json())
    current = await client.get(SNAPSHOT_CURRENT)
    next_snapshot = max(int(current or "0"), meta.snapshot_id)
    await client.put(SNAPSHOT_CURRENT, str(next_snapshot))
    return {"registered": True, "segment_id": meta.segment_id, "snapshot_id": next_snapshot}


@app.delete("/segments/{segment_id}")
async def delete_segment(segment_id: str) -> dict:
    if not leader:
        return await _proxy_to_leader("DELETE", f"/segments/{segment_id}", {})
    client = _require_etcd()
    await client.delete(f"{SEGMENT_PREFIX}{segment_id}")
    return {"deleted": True, "segment_id": segment_id}


@app.get("/health")
async def health() -> dict:
    await _sync_leader_state()
    await _refresh_nodes()
    return {
        "leader": leader,
        "ring_size": ring.ring_size,
        "search_fanout_mode": settings.search_fanout_mode,
        "nodes": [node.model_dump() for node in nodes.values()],
    }


async def _register_self() -> None:
    client = _require_etcd()
    address = _node_address(settings.node_id)
    info = NodeInfo(
        node_id=settings.node_id,
        role=settings.node_role,
        address=address,
        shard_ids=[],
    )
    await client.put(f"{NODE_PREFIX}{settings.node_id}", info.model_dump_json(), ttl=10)


async def _refresh_nodes() -> None:
    client = _require_etcd()
    values = await client.get_prefix(NODE_PREFIX)
    nodes.clear()
    for value in values.values():
        info = NodeInfo.model_validate_json(value)
        nodes[info.node_id] = info
    _rebuild_ring()


async def _heartbeat_loop() -> None:
    while True:
        await _register_self()
        await _sync_leader_state()
        await asyncio.sleep(5)


async def _sync_leader_state() -> None:
    global leader
    client = _require_etcd()
    current = await client.get("/vecscaledb/leader")
    if leader:
        if current == settings.node_id:
            await client.put("/vecscaledb/elections/coordinator", settings.node_id, ttl=10)
            await client.put("/vecscaledb/leader", settings.node_id, ttl=10)
        else:
            leader = False
    elif current is None:
        leader = await client.campaign("coordinator", settings.node_id)


def _rebuild_ring() -> None:
    ring.clear()
    for node in _reader_nodes():
        ring.add_node(node.node_id)


def _reader_nodes() -> list[NodeInfo]:
    return sorted(
        [node for node in nodes.values() if node.role == "reader"],
        key=lambda node: node.node_id,
    )


def _find_writer() -> NodeInfo | None:
    writers = sorted(
        [node for node in nodes.values() if node.role == "writer"],
        key=lambda node: node.node_id,
    )
    return writers[0] if writers else None


def _search_targets(readers: list[NodeInfo]) -> list[NodeInfo]:
    """Return readers that should execute one search request.

    ``scatter`` preserves the Phase 5 behavior: every reader receives the query
    and the coordinator merges partial shard results.

    ``replicated`` is the Phase 6 benchmark mode: all readers have a full copy
    of the segment set, so each query is sent to exactly one reader in
    round-robin order. That lets QPS scale with the reader count instead of
    duplicating work on every node.
    """
    global search_counter
    mode = settings.search_fanout_mode.lower()
    if mode == "scatter" or len(readers) <= 1:
        return readers
    if mode != "replicated":
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported search fanout mode: {settings.search_fanout_mode}",
        )
    index = search_counter % len(readers)
    search_counter += 1
    return [readers[index]]


def _node_address(node_id: str) -> str:
    if settings.node_role == "coordinator":
        return f"http://{node_id}:8000"
    if settings.node_role == "writer":
        return settings.writer_url
    return f"http://{node_id}:8200"


def _require_etcd() -> EtcdClient:
    if etcd is None:
        raise HTTPException(status_code=503, detail="Coordinator metadata store is not ready.")
    return etcd


async def _proxy_to_leader(method: str, path: str, json_body: dict) -> dict:
    client = _require_etcd()
    leader_id = await client.get("/vecscaledb/leader")
    if leader_id is None or leader_id == settings.node_id:
        raise HTTPException(status_code=503, detail="No active coordinator leader.")
    await _refresh_nodes()
    leader_info = nodes.get(leader_id)
    if leader_info is None:
        raise HTTPException(status_code=503, detail="Leader is not registered.")

    async with httpx.AsyncClient(timeout=5.0) as http_client:
        response = await http_client.request(
            method,
            f"{leader_info.address}{path}",
            json=json_body if method != "DELETE" else None,
        )
        response.raise_for_status()
        return response.json()


async def _search_reader(
    client: httpx.AsyncClient,
    reader: NodeInfo,
    req: SearchRequest,
    snapshot_id: int,
) -> dict:
    response = await client.post(
        f"{reader.address}/search",
        json={**req.model_dump(), "snapshot_id": snapshot_id},
    )
    response.raise_for_status()
    return response.json()


async def _current_snapshot_id(client: EtcdClient) -> int:
    current = int(await client.get(SNAPSHOT_CURRENT) or "0")
    if current > 0:
        return current

    values = await client.get_prefix(SEGMENT_PREFIX)
    if not values:
        return current
    segments = [SegmentMeta.model_validate_json(value) for value in values.values()]
    return max(segment.snapshot_id for segment in segments)


def _metrics_gauges() -> dict:
    reader_count = len(_reader_nodes())
    return {
        "ring_size": ring.ring_size,
        "registered_nodes": len(nodes),
        "registered_readers": reader_count,
        "coordinator_leader": leader,
    }


if __name__ == "__main__":
    uvicorn.run(
        "vecscaledb.coordinator.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
