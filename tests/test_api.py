"""API endpoint integration tests.

Mocks chromadb and sets required env vars before importing app modules.
Uses an in-memory SQLite database for ticket persistence tests.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

# Set required env vars before any app imports
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Mock chromadb to avoid pydantic v1 issues with Python 3.14
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.dependencies import get_db_session
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import health, knowledge_base, tickets
from src.db.models import Base


@pytest.fixture
def test_app(tmp_path):
    """Create a standalone test FastAPI app with temp file DB."""
    db_path = tmp_path / "test.db"

    # Create tables synchronously first
    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    # Async engine for the app
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    app = FastAPI(
        title="AI Ticket Router",
        version="0.3.0",
    )
    app.add_middleware(RequestLoggingMiddleware)

    # Set up mocked app state
    mock_vectorstore = AsyncMock()
    mock_vectorstore.collection = Mock()
    mock_vectorstore.collection.count.return_value = 42
    mock_vectorstore.search = AsyncMock(return_value=[])
    mock_vectorstore.get_stats = AsyncMock(return_value={
        "total_documents": 5,
        "total_chunks": 87,
        "categories": {"billing": 18},
    })
    app.state.vectorstore = mock_vectorstore

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "intent": "billing",
        "sentiment": "neutral",
        "urgency": "medium",
        "action": "auto_respond",
        "auto_response": "Here is your answer.",
        "kb_confidence": 0.85,
        "agent_trace": ["classify_intent: billing"],
    })
    app.state.agent_graph = mock_graph

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    app.include_router(health.router)
    app.include_router(tickets.router)
    app.include_router(knowledge_base.router)

    return app


@pytest.fixture
def client(test_app):
    """Test client for the test app."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "chromadb" in data
        assert "timestamp" in data


class TestTicketHistory:
    """Tests for the ticket history endpoint."""

    def test_empty_history(self, client):
        """Empty database should return empty list."""
        response = client.get("/api/v1/tickets/history")
        assert response.status_code == 200
        data = response.json()
        assert data["tickets"] == []
        assert data["total"] == 0
        assert data["limit"] == 20
        assert data["offset"] == 0

    def test_history_after_processing(self, client):
        """Processing a ticket should persist it and make it appear in history."""
        client.post("/api/v1/tickets/process", json={
            "message": "Help with billing",
            "customer_id": "test_123",
            "channel": "web",
        })

        response = client.get("/api/v1/tickets/history")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tickets"][0]["intent"] == "billing"
        assert data["tickets"][0]["action"] == "auto_respond"
        assert data["tickets"][0]["ticket_id"].startswith("tkt_")

    def test_history_pagination(self, client):
        """History should support limit and offset."""
        response = client.get("/api/v1/tickets/history?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 0

    def test_history_intent_filter(self, client):
        """History should support intent filtering."""
        response = client.get("/api/v1/tickets/history?intent=billing")
        assert response.status_code == 200

    def test_history_invalid_limit(self, client):
        """Invalid limit should return 422."""
        response = client.get("/api/v1/tickets/history?limit=0")
        assert response.status_code == 422

    def test_history_invalid_intent(self, client):
        """Invalid intent should return 422."""
        response = client.get("/api/v1/tickets/history?intent=invalid_type")
        assert response.status_code == 422

    def test_history_multiple_tickets(self, client):
        """Multiple processed tickets should all be persisted in history."""
        for _ in range(3):
            client.post("/api/v1/tickets/process", json={
                "message": "Help with billing",
            })

        response = client.get("/api/v1/tickets/history")
        assert response.json()["total"] == 3


class TestProcessTicket:
    """Tests for the ticket processing endpoint."""

    def test_process_ticket_success(self, client):
        """Processing a valid ticket should return structured response."""
        response = client.post("/api/v1/tickets/process", json={
            "message": "I need help with billing",
            "customer_id": "cust_123",
            "channel": "email",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "auto_respond"
        assert data["classification"]["intent"] == "billing"
        assert data["ticket_id"].startswith("tkt_")
        assert data["processing_time_ms"] > 0
        assert "X-Request-ID" in response.headers

    def test_process_ticket_empty_message(self, client):
        """Empty message should return 422."""
        response = client.post("/api/v1/tickets/process", json={
            "message": "",
        })
        assert response.status_code == 422

    def test_process_ticket_minimal(self, client):
        """Should work with only required fields."""
        response = client.post("/api/v1/tickets/process", json={
            "message": "Help me",
        })
        assert response.status_code == 200

    def test_process_ticket_invalid_channel(self, client):
        """Invalid channel should return 422."""
        response = client.post("/api/v1/tickets/process", json={
            "message": "Help",
            "channel": "fax",
        })
        assert response.status_code == 422


class TestKnowledgeBase:
    """Tests for the knowledge base endpoints."""

    def test_search_kb(self, client):
        """KB search should return results."""
        response = client.get("/api/v1/knowledge-base/search?q=billing question")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "billing question"
        assert "results" in data

    def test_search_kb_short_query(self, client):
        """Short query should return 400."""
        response = client.get("/api/v1/knowledge-base/search?q=a")
        assert response.status_code == 400

    def test_search_kb_invalid_limit(self, client):
        """Invalid limit should return 400."""
        response = client.get("/api/v1/knowledge-base/search?q=test&limit=50")
        assert response.status_code == 400

    def test_search_kb_with_category(self, client):
        """KB search with category filter should work."""
        response = client.get("/api/v1/knowledge-base/search?q=billing&category=billing")
        assert response.status_code == 200

    def test_kb_stats(self, client):
        """KB stats should return statistics."""
        response = client.get("/api/v1/knowledge-base/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "categories" in data


class TestSwaggerDocs:
    """Tests for API documentation."""

    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "AI Ticket Router"

    def test_swagger_ui_available(self, client):
        """Swagger UI should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_ticket_history_in_schema(self, client):
        """Ticket history endpoint should be in the OpenAPI schema."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "/api/v1/tickets/history" in schema["paths"]

    def test_ticket_process_in_schema(self, client):
        """Ticket process endpoint should be in the OpenAPI schema."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "/api/v1/tickets/process" in schema["paths"]
