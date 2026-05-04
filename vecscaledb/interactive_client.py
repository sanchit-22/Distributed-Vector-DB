"""Interactive HTTP client for manually exercising VecScaleDB APIs.

Run from the project root:

    python -m vecscaledb.interactive_client

The script prompts for the target host/port and operation, then sends the
corresponding request to a single-node, writer, coordinator, or reader service.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS = {
    "insert": 8100,
    "delete": 8100,
    "search": 8000,
    "health": 8000,
    "segments": 8000,
    "ready": 8200,
    "metrics": 8000,
}


def main() -> None:
    print("VecScaleDB Interactive Client")
    print("----------------------------")
    print("Operations: insert, search, delete, health, segments, ready, metrics, quit")
    print("Common ports: coordinator=8000, writer=8100, reader=8200, single-node=8000")

    while True:
        operation = _prompt("Operation", default="search").strip().lower()
        if operation in {"q", "quit", "exit"}:
            print("Bye.")
            return

        if operation not in DEFAULT_PORTS:
            print(f"Unsupported operation: {operation}")
            continue

        host = _prompt("Host", default=DEFAULT_HOST)
        port = _prompt_int("Port", default=DEFAULT_PORTS[operation])
        base_url = f"http://{host}:{port}"

        try:
            response = _send_operation(base_url, operation)
            _print_response(response)
        except httpx.HTTPError as exc:
            print(f"HTTP error: {exc}")
        except ValueError as exc:
            print(f"Input error: {exc}")


def _send_operation(base_url: str, operation: str) -> httpx.Response:
    timeout = httpx.Timeout(60.0, connect=5.0)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        if operation == "insert":
            return client.post("/insert", json=_insert_body())
        if operation == "search":
            return client.post("/search", json=_search_body())
        if operation == "delete":
            return client.post("/delete", json=_delete_body())
        if operation == "health":
            return client.get("/health")
        if operation == "segments":
            return client.get("/segments")
        if operation == "ready":
            return client.get("/ready")
        if operation == "metrics":
            return client.get("/metrics")
    raise ValueError(f"Unsupported operation: {operation}")


def _insert_body() -> dict[str, Any]:
    print("\nInsert mode")
    print("Enter one or more rows. Leave ID empty when done.")
    print("Vector format: comma-separated floats, for example: 1,0,0,...")

    ids: list[int] = []
    vectors: list[list[float]] = []
    expected_dim: int | None = None

    while True:
        raw_id = input("ID: ").strip()
        if raw_id == "":
            break
        vector_id = int(raw_id)
        vector = _prompt_vector("Vector")
        if expected_dim is None:
            expected_dim = len(vector)
        elif len(vector) != expected_dim:
            raise ValueError(
                f"All inserted vectors must have the same dimension. "
                f"Expected {expected_dim}, got {len(vector)}."
            )
        ids.append(vector_id)
        vectors.append(vector)

    if not ids:
        raise ValueError("At least one ID/vector pair is required.")
    return {"ids": ids, "vectors": vectors}


def _search_body() -> dict[str, Any]:
    print("\nSearch mode")
    query = _prompt_vector("Query vector")
    top_k = _prompt_int("top_k", default=5)
    nprobe = _prompt_int("nprobe", default=32)
    return {"query": query, "top_k": top_k, "nprobe": nprobe}


def _delete_body() -> dict[str, Any]:
    print("\nDelete mode")
    ids = _prompt_ids("IDs to delete")
    if not ids:
        raise ValueError("At least one ID is required.")
    return {"ids": ids}


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    if value == "" and default is not None:
        return default
    return value


def _prompt_int(label: str, default: int | None = None) -> int:
    raw = _prompt(label, str(default) if default is not None else None)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _prompt_vector(label: str) -> list[float]:
    raw = input(f"{label}: ").strip()
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"{label} must be a list of numbers.")
        return [float(value) for value in parsed]
    return _parse_float_csv(raw)


def _prompt_ids(label: str) -> list[int]:
    raw = input(f"{label} (comma-separated): ").strip()
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"{label} must be a list of integers.")
        return [int(value) for value in parsed]
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    if not raw:
        raise ValueError("Vector cannot be empty.")
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _print_response(response: httpx.Response) -> None:
    print(f"\nStatus: {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
    print()


if __name__ == "__main__":
    main()
