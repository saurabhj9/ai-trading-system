"""Integration tests for the monitoring metrics endpoint."""

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_metrics_endpoint_returns_base_and_performance_metrics():
    """Ensure /metrics returns base metrics along with performance data by default."""
    response = client.get("/api/v1/monitoring/metrics")

    assert response.status_code == 200

    payload = response.json()

    # Base metric fields should always be present
    for field in (
        "timestamp",
        "uptime_seconds",
        "requests_total",
        "errors_total",
        "average_request_duration_seconds",
        "error_rate",
    ):
        assert field in payload

    # Performance block should be included when include_performance defaults to True
    performance = payload.get("performance")
    assert isinstance(performance, dict)

    for field in (
        "llm_calls_total",
        "llm_calls_avg_duration_seconds",
        "data_operations_total",
        "data_operations_avg_duration_seconds",
        "agent_operations_total",
        "agent_operations_avg_duration_seconds",
        "detailed_operations",
    ):
        assert field in performance


def test_metrics_endpoint_can_exclude_performance_block():
    """Verify include_performance flag removes performance metrics from the response."""
    response = client.get(
        "/api/v1/monitoring/metrics",
        params={"include_performance": "false"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert "performance" not in payload
