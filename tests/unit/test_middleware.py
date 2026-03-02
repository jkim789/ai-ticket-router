"""Tests for request logging middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import RequestLoggingMiddleware


@pytest.fixture
def middleware_app():
    """Create a test FastAPI app with the logging middleware."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("test error")

    return app


@pytest.fixture
def middleware_client(middleware_app):
    """Test client for the middleware app."""
    return TestClient(middleware_app, raise_server_exceptions=False)


def test_request_id_header(middleware_client):
    """Response should include X-Request-ID header."""
    response = middleware_client.get("/test")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 8


def test_processing_time_header(middleware_client):
    """Response should include X-Processing-Time-Ms header."""
    response = middleware_client.get("/test")
    assert "X-Processing-Time-Ms" in response.headers
    time_ms = float(response.headers["X-Processing-Time-Ms"])
    assert time_ms >= 0


def test_unique_request_ids(middleware_client):
    """Each request should get a unique request ID."""
    response1 = middleware_client.get("/test")
    response2 = middleware_client.get("/test")
    assert response1.headers["X-Request-ID"] != response2.headers["X-Request-ID"]


def test_successful_response_body(middleware_client):
    """Middleware should not alter the response body."""
    response = middleware_client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
