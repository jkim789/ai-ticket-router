"""
Pytest configuration and shared fixtures.

Provides reusable test fixtures for mocking LLM calls, database sessions,
and ChromaDB collections.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_groq_response():
    """Mock Groq API response for testing."""
    mock_choice = Mock()
    mock_choice.message.content = '{"intent": "billing", "sentiment": "neutral", "urgency": "medium", "reasoning": "Test"}'
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_vectorstore():
    """Mock VectorStore for testing."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[
        {
            "content": "Test KB article about billing",
            "score": 0.85,
            "metadata": {"category": "billing", "title": "Billing FAQ"}
        }
    ])
    mock.collection = Mock()
    return mock


@pytest.fixture
def sample_ticket_state():
    """Sample ticket state for testing."""
    return {
        "raw_message": "I need help with my billing",
        "customer_id": "test_123",
        "channel": "email",
        "timestamp": datetime.now(),
        "agent_trace": []
    }


@pytest.fixture
def mock_groq_evaluate_response():
    """Mock Groq API response for evaluate node (high confidence)."""
    mock_choice = Mock()
    mock_choice.message.content = '{"can_answer": true, "confidence": 0.9, "reasoning": "Articles directly address the question"}'
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_groq_evaluate_low_response():
    """Mock Groq API response for evaluate node (low confidence)."""
    mock_choice = Mock()
    mock_choice.message.content = '{"can_answer": false, "confidence": 0.4, "reasoning": "Articles do not address the question"}'
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_groq_respond_response():
    """Mock Groq API response for respond node."""
    mock_choice = Mock()
    mock_choice.message.content = "Thank you for reaching out! Based on our records, here is how to resolve your billing issue. Please let me know if you need further assistance."
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_groq_route_response():
    """Mock Groq API response for route node."""
    mock_choice = Mock()
    mock_choice.message.content = "Customer needs help with a billing dispute regarding order #12345. Requires human attention due to low KB confidence. No relevant KB articles available."
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def sample_ticket_state_with_kb():
    """Sample ticket state with KB results for testing."""
    return {
        "raw_message": "I need help with my billing",
        "customer_id": "test_123",
        "channel": "email",
        "timestamp": datetime.now(),
        "intent": "billing",
        "sentiment": "neutral",
        "urgency": "medium",
        "agent_trace": [],
        "kb_results": [
            {
                "content": "Test KB article about billing",
                "score": 0.85,
                "metadata": {"category": "billing", "title": "Billing FAQ"}
            },
            {
                "content": "Refund policy information",
                "score": 0.78,
                "metadata": {"category": "billing", "title": "Refund Policy"}
            }
        ]
    }


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.75")
