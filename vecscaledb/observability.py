"""Shared structured logging and lightweight metrics helpers."""

from __future__ import annotations

import inspect
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from vecscaledb.config import Settings


GaugeProvider = Callable[[], dict[str, int | float | bool] | Awaitable[dict[str, int | float | bool]]]


class MetricsRegistry:
    """In-process Prometheus text metrics for one service process."""

    def __init__(self, node_role: str, node_id: str) -> None:
        self.node_role = node_role
        self.node_id = node_id
        self.request_total: dict[tuple[str, str, int], int] = defaultdict(int)
        self.duration_count: dict[tuple[str, str], int] = defaultdict(int)
        self.duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self.search_count = 0
        self.search_duration_sum = 0.0

    def observe(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        self.request_total[(method, path, status_code)] += 1
        self.duration_count[(method, path)] += 1
        self.duration_sum[(method, path)] += duration_seconds
        if path == "/search":
            self.search_count += 1
            self.search_duration_sum += duration_seconds

    def render(self, gauges: dict[str, int | float | bool] | None = None) -> str:
        lines = [
            "# HELP vecscaledb_request_total Total HTTP requests.",
            "# TYPE vecscaledb_request_total counter",
        ]
        for (method, path, status_code), value in sorted(self.request_total.items()):
            lines.append(
                "vecscaledb_request_total"
                f'{{node_role="{_esc(self.node_role)}",node_id="{_esc(self.node_id)}",'
                f'method="{_esc(method)}",path="{_esc(path)}",status="{status_code}"}} {value}'
            )

        lines.extend(
            [
                "# HELP vecscaledb_request_duration_seconds HTTP request duration.",
                "# TYPE vecscaledb_request_duration_seconds summary",
            ]
        )
        for (method, path), count in sorted(self.duration_count.items()):
            label = (
                f'node_role="{_esc(self.node_role)}",node_id="{_esc(self.node_id)}",'
                f'method="{_esc(method)}",path="{_esc(path)}"'
            )
            lines.append(
                f"vecscaledb_request_duration_seconds_count{{{label}}} {count}"
            )
            lines.append(
                "vecscaledb_request_duration_seconds_sum"
                f"{{{label}}} {self.duration_sum[(method, path)]:.9f}"
            )

        lines.extend(
            [
                "# HELP vecscaledb_search_duration_seconds Search request duration.",
                "# TYPE vecscaledb_search_duration_seconds summary",
                "vecscaledb_search_duration_seconds_count"
                f'{{node_role="{_esc(self.node_role)}",node_id="{_esc(self.node_id)}"}} {self.search_count}',
                "vecscaledb_search_duration_seconds_sum"
                f'{{node_role="{_esc(self.node_role)}",node_id="{_esc(self.node_id)}"}} {self.search_duration_sum:.9f}',
            ]
        )

        for name, value in sorted((gauges or {}).items()):
            metric = f"vecscaledb_{name}"
            lines.append(f"# TYPE {metric} gauge")
            lines.append(
                f'{metric}{{node_role="{_esc(self.node_role)}",node_id="{_esc(self.node_id)}"}} {_num(value)}'
            )
        return "\n".join(lines) + "\n"


def configure_logging(settings: Settings) -> structlog.stdlib.BoundLogger:
    """Configure JSON logging and return a logger bound to node identity."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    return structlog.get_logger().bind(
        node_role=settings.node_role,
        node_id=settings.node_id,
    )


def attach_observability(
    app: FastAPI,
    settings: Settings,
    logger: structlog.stdlib.BoundLogger,
    gauge_provider: GaugeProvider | None = None,
) -> MetricsRegistry:
    """Attach request logging middleware and a Prometheus-style metrics route."""
    registry = MetricsRegistry(settings.node_role, settings.node_id)

    @app.middleware("http")
    async def _request_observability(request: Request, call_next: Callable[[Request], Any]):
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
        method = request.method
        path = request.url.path
        started = time.perf_counter()
        logger.info(
            "request.start",
            trace_id=trace_id,
            method=method,
            path=path,
        )
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            logger.exception(
                "request.error",
                trace_id=trace_id,
                method=method,
                path=path,
                error=str(exc),
            )
            raise
        finally:
            duration_seconds = time.perf_counter() - started
            registry.observe(method, path, status_code, duration_seconds)
            logger.info(
                "request.end",
                trace_id=trace_id,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_seconds * 1000.0,
            )

    @app.middleware("http")
    async def _trace_header(request: Request, call_next: Callable[[Request], Any]):
        response = await call_next(request)
        trace_id = getattr(request.state, "trace_id", None)
        if trace_id is not None:
            response.headers["x-trace-id"] = trace_id
        return response

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        gauges = await _collect_gauges(gauge_provider)
        return PlainTextResponse(registry.render(gauges))

    return registry


async def _collect_gauges(
    provider: GaugeProvider | None,
) -> dict[str, int | float | bool]:
    if provider is None:
        return {}
    values = provider()
    if inspect.isawaitable(values):
        values = await values
    return dict(values)


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _num(value: int | float | bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)
