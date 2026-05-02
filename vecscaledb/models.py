"""Pydantic schemas shared across all nodes."""

from pydantic import BaseModel, Field


class InsertRequest(BaseModel):
    vectors: list[list[float]] = Field(min_length=1)  # shape [N, dim]
    ids: list[int] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: list[float] = Field(min_length=1)  # shape [dim]
    top_k: int = Field(default=10, ge=1)
    nprobe: int = Field(default=32, ge=1)


class ReaderSearchRequest(SearchRequest):
    snapshot_id: int = Field(default=0, ge=0)


class DeleteRequest(BaseModel):
    ids: list[int] = Field(min_length=1)


class SearchResult(BaseModel):
    ids: list[int]
    distances: list[float]


class SegmentMeta(BaseModel):
    segment_id: str = Field(min_length=1)
    shard_id: int = Field(ge=0)
    num_vectors: int = Field(ge=0)
    snapshot_id: int = Field(ge=0)
    path: str = Field(min_length=1)  # absolute path inside shared_storage


class NodeInfo(BaseModel):
    node_id: str = Field(min_length=1)
    role: str = Field(min_length=1)  # coordinator | writer | reader
    address: str = Field(min_length=1)
    shard_ids: list[int] = []
