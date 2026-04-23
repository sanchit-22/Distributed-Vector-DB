"""Pydantic schemas shared across all nodes."""

from pydantic import BaseModel


class InsertRequest(BaseModel):
    vectors: list[list[float]]  # shape [N, dim]
    ids: list[int]


class SearchRequest(BaseModel):
    query: list[float]  # shape [dim]
    top_k: int = 10
    nprobe: int = 32


class SearchResult(BaseModel):
    ids: list[int]
    distances: list[float]


class SegmentMeta(BaseModel):
    segment_id: str
    shard_id: int
    num_vectors: int
    snapshot_id: int
    path: str  # absolute path inside shared_storage


class NodeInfo(BaseModel):
    node_id: str
    role: str  # coordinator | writer | reader
    address: str
    shard_ids: list[int] = []
