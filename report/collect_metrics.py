"""Collect local dataset and benchmark metrics for the PDF report.

This script does not run heavy benchmarks. It normalizes the benchmark outputs
that already exist on disk and records which datasets are available locally.

Run from the project root:

    python report/collect_metrics.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
BUILD_DIR = REPORT_DIR / "build"
RESULTS_DIR = REPORT_DIR / "results"


DATASETS = {
    "SIFT10K": {
        "vectors": 10_000,
        "dims": 128,
        "metric": "L2",
        "role": "Unit tests and correctness checks",
        "path": ROOT / "data" / "sift1m" / "sift1m.hdf5",
        "derived_from": "SIFT1M train[:10000]",
    },
    "SIFT1M": {
        "vectors": 1_000_000,
        "dims": 128,
        "metric": "L2",
        "role": "Primary performance benchmark",
        "path": ROOT / "data" / "sift1m" / "sift1m.hdf5",
        "derived_from": None,
    },
    "Deep1M": {
        "vectors": 1_000_000,
        "dims": 96,
        "metric": "L2",
        "role": "Paper replication dataset",
        "path": ROOT / "data" / "deep1m" / "deep1m.hdf5",
        "derived_from": None,
    },
}


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": "VecScaleDB",
            "summary": (
                "Distributed vector database with writer-side WAL/MemTable "
                "ingestion, immutable Faiss IVF_FLAT segments, coordinator "
                "metadata, and replicated or scatter-gather readers."
            ),
        },
        "datasets": collect_datasets(),
        "benchmarks": collect_benchmarks(),
        "notes": [
            "SIFT10K is derived as the first 10,000 training vectors from SIFT1M.",
            "SIFT1M local scaling metrics are read from scaling_results.json when present.",
            "Deep1M is included in the comparison table but is marked pending unless data and results are supplied locally.",
        ],
    }

    out_path = BUILD_DIR / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


def collect_datasets() -> list[dict[str, Any]]:
    rows = []
    h5_info = _hdf5_info_available()
    for name, spec in DATASETS.items():
        path = spec["path"]
        exists = path.exists()
        actual = _inspect_hdf5(path) if exists and h5_info else {}
        rows.append(
            {
                "name": name,
                "expected_vectors": spec["vectors"],
                "expected_dims": spec["dims"],
                "metric": spec["metric"],
                "role": spec["role"],
                "path": str(path.relative_to(ROOT)),
                "available": exists,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if exists else 0,
                "derived_from": spec["derived_from"],
                "hdf5_keys": actual.get("keys", {}),
            }
        )
    return rows


def collect_benchmarks() -> dict[str, Any]:
    dataset_metrics = {
        "SIFT10K": _load_json(RESULTS_DIR / "sift10k_metrics.json") or {},
        "SIFT1M": _load_json(RESULTS_DIR / "sift1m_metrics.json") or {},
        "Deep1M": _load_json(RESULTS_DIR / "deep1m_metrics.json") or {},
    }
    scaling = {
        "SIFT10K": dataset_metrics["SIFT10K"].get("scaling")
        or _load_json(RESULTS_DIR / "sift10k_scaling_results.json"),
        "SIFT1M": dataset_metrics["SIFT1M"].get("scaling")
        or _load_json(RESULTS_DIR / "sift1m_scaling_results.json")
        or _load_json(ROOT / "scaling_results.json"),
        "Deep1M": dataset_metrics["Deep1M"].get("scaling")
        or _load_json(RESULTS_DIR / "deep1m_scaling_results.json"),
    }
    qps = {
        name: value.get("qps", {})
        for name, value in dataset_metrics.items()
    }
    recall = {
        name: value.get("recall", {})
        for name, value in dataset_metrics.items()
        if value.get("recall")
    }
    load = {
        name: value.get("load", {})
        for name, value in dataset_metrics.items()
        if value.get("load")
    }
    return {
        "dataset_metrics": dataset_metrics,
        "scaling": {
            name: _normalize_scaling(value)
            for name, value in scaling.items()
        },
        "qps": qps,
        "recall": recall,
        "load": load,
    }


def _normalize_scaling(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    rows = []
    for row in data:
        if not isinstance(row, dict):
            continue
        readers = _as_int(row.get("readers"))
        qps = _as_float(row.get("qps"))
        if readers is None or qps is None:
            continue
        rows.append(
            {
                "readers": readers,
                "qps": qps,
                "p50_ms": _as_float(row.get("p50_ms")) or 0.0,
                "p95_ms": _as_float(row.get("p95_ms")) or 0.0,
                "p99_ms": _as_float(row.get("p99_ms")) or 0.0,
                "errors": _as_int(row.get("errors")) or 0,
                "total_queries": _as_int(row.get("total_queries")) or 0,
                "elapsed_seconds": _as_float(row.get("elapsed_seconds")) or 0.0,
            }
        )
    return sorted(rows, key=lambda item: item["readers"])


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _hdf5_info_available() -> bool:
    try:
        import h5py  # noqa: F401
    except Exception:
        return False
    return True


def _inspect_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except Exception:
        return {}
    keys = {}
    try:
        with h5py.File(path, "r") as h5:
            for key in h5.keys():
                dataset = h5[key]
                keys[key] = {
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                }
    except OSError:
        return {}
    return {"keys": keys}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
