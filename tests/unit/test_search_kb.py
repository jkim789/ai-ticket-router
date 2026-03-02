"""Unit tests for knowledge base search node."""

from unittest.mock import AsyncMock

import pytest

from src.agent.nodes.search_kb import search_knowledge_base


@pytest.mark.asyncio
async def test_search_kb_with_results(sample_ticket_state, mock_vectorstore):
    """Test successful KB search returns formatted results."""
    sample_ticket_state["intent"] = "billing"

    result = await search_knowledge_base(sample_ticket_state, vectorstore=mock_vectorstore)

    assert len(result["kb_results"]) == 1
    assert result["kb_results"][0]["content"] == "Test KB article about billing"
    assert result["kb_results"][0]["score"] == 0.85
    assert "search_kb" in result["agent_trace"][0]
    assert "1 results" in result["agent_trace"][0]
    mock_vectorstore.search.assert_called_once_with(
        query="I need help with my billing",
        category="billing",
        limit=5,
    )


@pytest.mark.asyncio
async def test_search_kb_no_results(sample_ticket_state):
    """Test KB search with no matching documents."""
    sample_ticket_state["intent"] = "general"
    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(return_value=[])

    result = await search_knowledge_base(sample_ticket_state, vectorstore=mock_vs)

    assert result["kb_results"] == []
    assert "0 results" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_search_kb_error_handling(sample_ticket_state):
    """Test KB search handles vectorstore errors gracefully."""
    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(side_effect=Exception("ChromaDB unavailable"))

    result = await search_knowledge_base(sample_ticket_state, vectorstore=mock_vs)

    assert result["kb_results"] == []
    assert "ERROR" in result["agent_trace"][0]
    assert "ChromaDB unavailable" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_search_kb_preserves_metadata(sample_ticket_state):
    """Test that search results preserve document metadata."""
    sample_ticket_state["intent"] = "technical"
    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(return_value=[
        {
            "content": "How to reset your password",
            "score": 0.92,
            "metadata": {"category": "technical", "title": "Password Reset"}
        },
        {
            "content": "Account setup guide",
            "score": 0.71,
            "metadata": {"category": "technical", "title": "Getting Started"}
        }
    ])

    result = await search_knowledge_base(sample_ticket_state, vectorstore=mock_vs)

    assert len(result["kb_results"]) == 2
    assert result["kb_results"][0]["metadata"]["title"] == "Password Reset"
    assert result["kb_results"][1]["score"] == 0.71
    assert "top score 0.920" in result["agent_trace"][0]
