"""Feature-level tests rebuilt from the VecScaleDB project proposal.

The tests in this file intentionally map to the features listed in
``VecScaleDB.pdf``: IVF_FLAT search, dynamic writes/deletes, snapshot isolation,
shared-storage reader recovery, scatter-gather routing, coordinator HA, and the
multi-reader evaluation path.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import time

import httpx
import numpy as np
import pytest
from fastapi import HTTPException

from vecscaledb.config import Settings
from vecscaledb.coordinator.etcd_client import EtcdClient, _GLOBAL_MEMORY_STORE
from vecscaledb.coordinator.ring import ConsistentHashRing
from vecscaledb.index.ivf_flat import IVFFlatIndex
from vecscaledb.index.segment import Segment
from vecscaledb.models import (
    InsertRequest,
    NodeInfo,
    ReaderSearchRequest,
    SearchRequest,
    SegmentMeta,
)
from vecscaledb.reader.search import SegmentCache
from vecscaledb.storage.lsm import LSMEngine, merge_results
from vecscaledb.storage.wal import OP_DELETE, OP_INSERT, WAL
from vecscaledb.writer.pipeline import WriterPipeline


DIM = 8
NLIST = 2


class RecordingCoordinator:
    def __init__(self) -> None:
        self.nodes: list[NodeInfo] = []
        self.segments: list[SegmentMeta] = []
        self.deleted_segments: list[str] = []

    async def register_node(self, info: NodeInfo) -> None:
        self.nodes.append(info)

    async def register_segment(self, meta: SegmentMeta) -> None:
        if meta.segment_id not in {segment.segment_id for segment in self.segments}:
            self.segments.append(meta)

    async def delete_segment(self, segment_id: str) -> None:
        self.deleted_segments.append(segment_id)
        self.segments = [
            segment for segment in self.segments if segment.segment_id != segment_id
        ]

    async def get_all_segments(self) -> list[SegmentMeta]:
        return list(self.segments)


class StaticCoordinator:
    def __init__(self, segments: list[SegmentMeta]) -> None:
        self.segments = segments
        self.nodes: list[NodeInfo] = []

    async def register_node(self, info: NodeInfo) -> None:
        self.nodes.append(info)

    async def get_all_segments(self) -> list[SegmentMeta]:
        return list(self.segments)


def basis(count: int, dim: int = DIM) -> np.ndarray:
    vectors = np.zeros((count, dim), dtype=np.float32)
    for row in range(count):
        vectors[row, row % dim] = 1.0
        vectors[row, (row + 1) % dim] += row * 0.01
    return vectors


def make_engine(
    tmp_path,
    *,
    shard_id: int = 0,
    flush_threshold: int = 100,
    merge_threshold: int = 4,
) -> LSMEngine:
    return LSMEngine(
        shared_storage=str(tmp_path / "shared"),
        wal_path=str(tmp_path / f"wal-{shard_id}"),
        shard_id=shard_id,
        dim=DIM,
        nlist=NLIST,
        flush_threshold=flush_threshold,
        merge_threshold=merge_threshold,
    )


def make_writer_settings(tmp_path, *, num_shards: int = 1, flush_threshold: int = 100):
    return Settings(
        node_role="writer",
        node_id="writer-0",
        writer_url="http://writer-0:8100",
        coordinator_urls=["http://coordinator-0:8000"],
        shared_storage_path=str(tmp_path / "shared"),
        wal_path=str(tmp_path / "wal"),
        dim=DIM,
        nlist=NLIST,
        nprobe=NLIST,
        num_shards=num_shards,
        segment_flush_threshold=flush_threshold,
        segment_merge_threshold=10,
    )


def create_segment(tmp_path, shard_id: int, snapshot_id: int, ids: list[int]) -> Segment:
    return Segment.create(
        shard_id=shard_id,
        snapshot_id=snapshot_id,
        vectors=basis(len(ids)),
        ids=np.array(ids, dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "shared" / "segments"),
    )


def test_ivf_flat_search_uses_external_ids_nprobe_and_persistence(tmp_path):
    vectors = basis(12)
    external_ids = np.arange(100, 112, dtype=np.int64)

    index = IVFFlatIndex(dim=DIM, nlist=NLIST)
    index.train(vectors)
    index.add(vectors, external_ids)

    distances, ids = index.search(vectors[4], top_k=3, nprobe=NLIST)
    assert ids[0] == 104
    assert distances[0] == pytest.approx(0.0)
    assert index.ntotal == len(external_ids)

    saved = tmp_path / "index.faiss"
    index.save(str(saved))
    loaded = IVFFlatIndex(dim=DIM, nlist=NLIST)
    loaded.load(str(saved))

    loaded_distances, loaded_ids = loaded.search(vectors[4], top_k=3, nprobe=NLIST)
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_allclose(loaded_distances, distances)


def test_wal_recovers_valid_records_and_ignores_corrupt_tail(tmp_path):
    wal = WAL(str(tmp_path / "wal"), dim=DIM)
    wal.open()
    wal.append(OP_INSERT, np.array([1, 2]), basis(2))
    wal.append(OP_DELETE, np.array([1]), np.zeros((1, DIM), dtype=np.float32))
    wal.close()

    with open(wal.file_path, "r+b") as handle:
        handle.seek(-4, os.SEEK_END)
        handle.write(b"\x00\x00\x00\x00")

    records = WAL(str(tmp_path / "wal"), dim=DIM).recover()

    assert [record.lsn for record in records] == [1]
    assert records[0].op_type == OP_INSERT
    np.testing.assert_array_equal(records[0].ids, np.array([1, 2]))


def test_segment_persists_metadata_arrays_and_searchable_index(tmp_path):
    vectors = basis(3)
    segment = Segment.create(
        shard_id=2,
        snapshot_id=9,
        vectors=vectors,
        ids=np.array([20, 21, 22], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "shared" / "segments"),
    )

    assert os.path.isfile(os.path.join(segment.path, "index.faiss"))
    assert os.path.isfile(os.path.join(segment.path, "ids.npy"))
    assert os.path.isfile(os.path.join(segment.path, "vectors.npy"))
    assert segment.meta.shard_id == 2
    assert segment.meta.snapshot_id == 9

    loaded = Segment.load(segment.path)
    distances, ids = loaded.search(vectors[1], top_k=1, nprobe=NLIST)

    assert ids.tolist() == [21]
    assert distances.tolist() == pytest.approx([0.0])


@pytest.mark.asyncio
async def test_lsm_insert_search_flush_and_wal_crash_recovery(tmp_path):
    first = make_engine(tmp_path, flush_threshold=10)
    await first.start()
    await first.insert(np.array([10, 11], dtype=np.int64), basis(2))

    first.wal.close()
    first._started = False

    recovered = make_engine(tmp_path, flush_threshold=10)
    await recovered.start()

    health = await recovered.health()
    ids, distances = await recovered.search(basis(2)[1], top_k=1, nprobe=NLIST)

    assert health["memtable_size"] == 2
    assert ids == [11]
    assert distances[0] == pytest.approx(0.0)
    await recovered.stop()


@pytest.mark.asyncio
async def test_lsm_delete_tombstone_survives_flush_restart_and_health(tmp_path):
    engine = make_engine(tmp_path, flush_threshold=2)
    await engine.start()
    vectors = basis(3)
    await engine.insert(np.array([1, 2, 3], dtype=np.int64), vectors)
    await engine.delete(np.array([2], dtype=np.int64))

    health = await engine.health()
    ids, _ = await engine.search(vectors[1], top_k=3, nprobe=NLIST)

    assert health["ntotal"] == 2
    assert 2 not in ids
    await engine.stop()

    restarted = make_engine(tmp_path, flush_threshold=2)
    await restarted.start()
    ids_after_restart, _ = await restarted.search(vectors[1], top_k=3, nprobe=NLIST)

    assert 2 not in ids_after_restart
    assert (tmp_path / "shared" / "tombstones.json").is_file()
    await restarted.stop()


@pytest.mark.asyncio
async def test_snapshot_isolation_hides_newer_writes_until_current_snapshot(tmp_path):
    engine = make_engine(tmp_path, flush_threshold=100)
    await engine.start()
    vectors = basis(2)

    first = await engine.insert(np.array([1], dtype=np.int64), vectors[:1])
    await engine.insert(np.array([2], dtype=np.int64), vectors[1:2])

    old_ids, _ = await engine.search(
        vectors[1], top_k=2, nprobe=NLIST, snapshot_id=first["lsn"]
    )
    current_ids, _ = await engine.search(vectors[1], top_k=1, nprobe=NLIST)

    assert 2 not in old_ids
    assert current_ids == [2]
    await engine.stop()


@pytest.mark.asyncio
async def test_duplicate_id_update_uses_latest_visible_vector(tmp_path):
    engine = make_engine(tmp_path, flush_threshold=100)
    await engine.start()
    old_vector = basis(1)
    new_vector = basis(2)[1:2]

    await engine.insert(np.array([7], dtype=np.int64), old_vector)
    await engine.flush()
    await engine.insert(np.array([7], dtype=np.int64), new_vector)

    ids_for_old_query, distances_for_old_query = await engine.search(
        old_vector[0], top_k=1, nprobe=NLIST
    )
    ids_for_new_query, distances_for_new_query = await engine.search(
        new_vector[0], top_k=1, nprobe=NLIST
    )

    assert ids_for_old_query == [7]
    assert distances_for_old_query[0] > 0.5
    assert ids_for_new_query == [7]
    assert distances_for_new_query[0] == pytest.approx(0.0)
    await engine.stop()


@pytest.mark.asyncio
async def test_reader_cache_supports_sharding_replicas_tombstones_and_lru(tmp_path):
    own = create_segment(tmp_path, shard_id=0, snapshot_id=1, ids=[10, 11])
    other = create_segment(tmp_path, shard_id=1, snapshot_id=1, ids=[20, 21])
    tombstone_path = tmp_path / "shared" / "tombstones.json"
    tombstone_path.write_text(json.dumps([11]))

    shard_cache = SegmentCache(str(tmp_path / "shared"), shard_id=0)
    await shard_cache.refresh([own.meta, other.meta])
    ids, _ = shard_cache.search_all(basis(2)[1], top_k=5, nprobe=NLIST, snapshot_id=9)

    assert shard_cache.segment_ids == [own.segment_id]
    assert shard_cache.ntotal == 1
    assert 11 not in ids

    replica_cache = SegmentCache(
        str(tmp_path / "shared"), shard_id=99, load_all_shards=True
    )
    await replica_cache.refresh([own.meta, other.meta])
    assert set(replica_cache.segment_ids) == {own.segment_id, other.segment_id}

    limited_cache = SegmentCache(
        str(tmp_path / "shared"), shard_id=99, load_all_shards=True, max_segments_in_memory=1
    )
    await limited_cache.refresh([own.meta, other.meta])
    assert limited_cache.segments_loaded == 1
    assert own.segment_id not in limited_cache.segment_ids


@pytest.mark.asyncio
async def test_reader_cache_reload_hides_stale_duplicate_segment_versions(tmp_path):
    old = Segment.create(
        shard_id=0,
        snapshot_id=1,
        vectors=basis(1),
        ids=np.array([70], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "shared" / "segments"),
    )
    new = Segment.create(
        shard_id=0,
        snapshot_id=2,
        vectors=basis(2)[1:2],
        ids=np.array([70], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "shared" / "segments"),
    )
    cache = SegmentCache(str(tmp_path / "shared"), shard_id=0)
    await cache.refresh([old.meta, new.meta])

    ids, distances = cache.search_all(basis(1)[0], top_k=1, nprobe=NLIST, snapshot_id=9)

    assert ids == [70]
    assert distances[0] > 0.5


@pytest.mark.asyncio
async def test_scatter_gather_search_merges_results_from_multiple_shards(tmp_path):
    vectors = basis(6)
    even = Segment.create(
        shard_id=0,
        snapshot_id=4,
        vectors=vectors[[0, 2, 4]],
        ids=np.array([0, 2, 4], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "shared" / "segments"),
    )
    odd = Segment.create(
        shard_id=1,
        snapshot_id=4,
        vectors=vectors[[1, 3, 5]],
        ids=np.array([1, 3, 5], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(tmp_path / "shared" / "segments"),
    )
    cache_even = SegmentCache(str(tmp_path / "shared"), shard_id=0)
    cache_odd = SegmentCache(str(tmp_path / "shared"), shard_id=1)
    await cache_even.refresh([even.meta, odd.meta])
    await cache_odd.refresh([even.meta, odd.meta])

    even_ids, even_distances = cache_even.search_all(
        vectors[0], top_k=6, nprobe=NLIST, snapshot_id=4
    )
    odd_ids, odd_distances = cache_odd.search_all(
        vectors[0], top_k=6, nprobe=NLIST, snapshot_id=4
    )
    merged_ids, _ = merge_results(
        [
            (np.array(even_distances, dtype=np.float32), np.array(even_ids)),
            (np.array(odd_distances, dtype=np.float32), np.array(odd_ids)),
        ],
        top_k=6,
    )

    assert merged_ids[0] == 0
    assert any(vector_id % 2 == 0 for vector_id in merged_ids)
    assert any(vector_id % 2 == 1 for vector_id in merged_ids)


@pytest.mark.asyncio
async def test_writer_pipeline_shards_flushes_registers_and_persists_deletes(tmp_path):
    coordinator = RecordingCoordinator()
    pipeline = WriterPipeline(
        make_writer_settings(tmp_path, num_shards=3, flush_threshold=1),
        coordinator_client=coordinator,
    )
    await pipeline.start()
    vectors = basis(3)

    result = await pipeline.insert(np.array([0, 1, 2], dtype=np.int64), vectors)
    delete_result = await pipeline.delete(np.array([1], dtype=np.int64))

    assert result["inserted"] == 3
    assert result["segment_flushed"] is True
    assert coordinator.nodes[-1].shard_ids == [0, 1, 2]
    assert {segment.shard_id for segment in coordinator.segments} == {0, 1, 2}
    assert delete_result["deleted"] == 1
    assert json.loads((tmp_path / "shared" / "tombstones.json").read_text()) == [1]
    await pipeline.stop()


@pytest.mark.asyncio
async def test_etcd_memory_backend_models_coordinator_ha_ttl_failover():
    EtcdClient.reset_memory()
    first = EtcdClient(["memory://local"])
    second = EtcdClient(["memory://local"])

    assert await first.campaign("coordinator", "coordinator-0") is True
    assert await second.campaign("coordinator", "coordinator-1") is False

    _GLOBAL_MEMORY_STORE.ttls["/vecscaledb/elections/coordinator"] = (
        time.monotonic() - 1
    )
    _GLOBAL_MEMORY_STORE.ttls["/vecscaledb/leader"] = time.monotonic() - 1

    assert await second.campaign("coordinator", "coordinator-1") is True
    assert await second.get("/vecscaledb/leader") == "coordinator-1"
    EtcdClient.reset_memory()


def test_consistent_hash_ring_routes_and_rebalances_reader_nodes():
    ring = ConsistentHashRing(virtual_nodes=5)
    for node_id in ["reader-0", "reader-1", "reader-2"]:
        ring.add_node(node_id)

    owner = ring.get_node("query-42")
    owners = ring.get_nodes_for_range(1, 50)
    ring.remove_node(owner)

    assert ring.ring_size == 10
    assert owner not in ring.nodes
    assert owners <= {"reader-0", "reader-1", "reader-2"}
    assert ring.get_node("query-42") in ring.nodes


def test_coordinator_client_fails_over_between_replicas(monkeypatch):
    from vecscaledb.coordinator.client import CoordinatorClient

    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, **kwargs):
            calls.append(url)
            if "coordinator-0" in url:
                raise httpx.ConnectError("down", request=httpx.Request(method, url))
            return httpx.Response(
                200,
                json={"registered": True},
                request=httpx.Request(method, url),
            )

    monkeypatch.setattr(
        "vecscaledb.coordinator.client.httpx.AsyncClient", FakeAsyncClient
    )
    client = CoordinatorClient(
        ["http://coordinator-0:8000", "http://coordinator-1:8000"]
    )

    asyncio.run(
        client.register_node(
            NodeInfo(
                node_id="reader-0",
                role="reader",
                address="http://reader-0:8200",
                shard_ids=[0],
            )
        )
    )

    assert calls == [
        "http://coordinator-0:8000/register",
        "http://coordinator-1:8000/register",
    ]


@pytest.mark.asyncio
async def test_single_node_api_validates_and_searches_flushed_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("VECSCALE_DIM", str(DIM))
    monkeypatch.setenv("VECSCALE_NLIST", str(NLIST))
    monkeypatch.setenv("VECSCALE_NPROBE", str(NLIST))
    monkeypatch.setenv("VECSCALE_SEGMENT_FLUSH_THRESHOLD", "2")
    monkeypatch.setenv("VECSCALE_SHARED_STORAGE_PATH", str(tmp_path / "shared"))
    monkeypatch.setenv("VECSCALE_WAL_PATH", str(tmp_path / "wal"))

    import vecscaledb.node as node_app

    node_app = importlib.reload(node_app)
    async with node_app.lifespan(node_app.app):
        insert = await node_app.insert(
            InsertRequest(ids=[1, 2], vectors=basis(2).tolist())
        )
        search = await node_app.search(
            SearchRequest(query=basis(2)[1].tolist(), top_k=1, nprobe=NLIST)
        )
        with pytest.raises(HTTPException) as bad_insert:
            await node_app.insert(InsertRequest(ids=[1], vectors=[[1.0]]))

    assert insert["segment_flushed"] is True
    assert search.ids == [2]
    assert bad_insert.value.status_code == 400


@pytest.mark.asyncio
async def test_reader_api_reloads_shared_storage_after_restart(monkeypatch, tmp_path):
    shared = tmp_path / "shared"
    segment = Segment.create(
        shard_id=0,
        snapshot_id=6,
        vectors=basis(2),
        ids=np.array([30, 31], dtype=np.int64),
        dim=DIM,
        nlist=NLIST,
        base_path=str(shared / "segments"),
    )

    monkeypatch.setenv("VECSCALE_NODE_ROLE", "reader")
    monkeypatch.setenv("VECSCALE_NODE_ID", "reader-0")
    monkeypatch.setenv("VECSCALE_READER_SHARD_ID", "0")
    monkeypatch.setenv("VECSCALE_DIM", str(DIM))
    monkeypatch.setenv("VECSCALE_NLIST", str(NLIST))
    monkeypatch.setenv("VECSCALE_SHARED_STORAGE_PATH", str(shared))
    monkeypatch.setenv("VECSCALE_COORDINATOR_URLS", '["http://coordinator-0:8000"]')

    import vecscaledb.reader.app as reader_app

    reader_app = importlib.reload(reader_app)
    coordinator = StaticCoordinator([segment.meta])
    reader_app.create_coordinator_client = lambda: coordinator
    reader_app.create_cache = lambda: SegmentCache(str(shared), shard_id=0)

    async with reader_app.lifespan(reader_app.app):
        first = await reader_app.search(
            ReaderSearchRequest(
                query=basis(2)[1].tolist(),
                top_k=1,
                nprobe=NLIST,
                snapshot_id=6,
            )
        )

    async with reader_app.lifespan(reader_app.app):
        restarted = await reader_app.search(
            ReaderSearchRequest(
                query=basis(2)[1].tolist(),
                top_k=1,
                nprobe=NLIST,
                snapshot_id=6,
            )
        )

    assert coordinator.nodes[0].role == "reader"
    assert first.ids == [31]
    assert restarted.ids == [31]
