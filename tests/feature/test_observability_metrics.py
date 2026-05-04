"""Direct observability tests that avoid ASGI TestClient threading."""

from vecscaledb.observability import MetricsRegistry


def test_metrics_registry_records_requests_search_latency_and_gauges():
    registry = MetricsRegistry(node_role="reader", node_id="reader-0")

    registry.observe("POST", "/search", 200, 0.125)
    registry.observe("GET", "/ready", 503, 0.025)
    rendered = registry.render({"segments_loaded": 2, "reader_ready": True})

    assert 'path="/search"' in rendered
    assert 'status="200"' in rendered
    assert "vecscaledb_search_duration_seconds_count" in rendered
    assert "vecscaledb_search_duration_seconds_sum" in rendered
    assert "vecscaledb_segments_loaded" in rendered
    assert "vecscaledb_reader_ready" in rendered
    assert 'node_role="reader"' in rendered
    assert 'node_id="reader-0"' in rendered
