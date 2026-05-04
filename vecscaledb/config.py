"""Central configuration driven by environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    node_role: str = "reader"  # coordinator | writer | reader
    node_id: str = "node-0"
    reader_shard_id: int = 0
    reader_load_all_shards: bool = False
    reader_max_segments_in_memory: int = 20
    search_fanout_mode: str = "scatter"  # scatter | replicated
    coordinator_urls: list[str] = ["http://coordinator-0:8000"]
    writer_url: str = "http://writer-0:8100"
    shared_storage_path: str = "/shared_storage"
    wal_path: str = "/wal"
    etcd_endpoints: list[str] = ["http://etcd:2379"]
    dim: int = 128
    nlist: int = 256  # IVF_FLAT number of centroids
    nprobe: int = 32  # IVF_FLAT search-time probe count
    segment_flush_threshold: int = 50_000  # vectors before flush
    segment_merge_threshold: int = 4  # segments before merge
    snapshot_ttl_seconds: int = 60
    num_shards: int = 1  # number of shards for scatter-gather
    coordinator_reader_timeout_seconds: float = 10.0

    class Config:
        env_prefix = "VECSCALE_"
