"""Central configuration driven by environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    node_role: str = "reader"  # coordinator | writer | reader
    node_id: str = "node-0"
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

    class Config:
        env_prefix = "VECSCALE_"
