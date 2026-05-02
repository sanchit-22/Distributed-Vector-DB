"""Tests for shared observability helpers."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vecscaledb.config import Settings
from vecscaledb.observability import attach_observability, configure_logging


def test_observability_adds_trace_header_metrics_and_json_logs(capsys):
    settings = Settings(node_role="reader", node_id="reader-test")
    logger = configure_logging(settings)
    app = FastAPI()
    attach_observability(
        app,
        settings,
        logger,
        lambda: {"segments_loaded": 2, "reader_ready": True},
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/ping", headers={"x-trace-id": "trace-123"})
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "trace-123"
    assert "vecscaledb_request_total" in metrics.text
    assert 'node_role="reader"' in metrics.text
    assert 'node_id="reader-test"' in metrics.text
    assert "vecscaledb_segments_loaded" in metrics.text
    assert "vecscaledb_reader_ready" in metrics.text

    log_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    request_logs = [line for line in log_lines if line.get("event") == "request.end"]
    assert request_logs
    assert request_logs[0]["trace_id"] == "trace-123"
    assert request_logs[0]["node_role"] == "reader"
    assert request_logs[0]["node_id"] == "reader-test"
