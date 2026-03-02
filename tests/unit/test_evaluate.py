"""Unit tests for confidence evaluation node."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.evaluate import evaluate_confidence


@pytest.mark.asyncio
async def test_evaluate_no_kb_results(sample_ticket_state):
    """Test evaluation with no KB results routes to human."""
    sample_ticket_state["kb_results"] = []

    result = await evaluate_confidence(sample_ticket_state)

    assert result["kb_confidence"] == 0.0
    assert result["action"] == "route_to_human"
    assert "no KB results" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_evaluate_low_similarity(sample_ticket_state):
    """Test low similarity scores route to human without LLM call."""
    sample_ticket_state["kb_results"] = [
        {"content": "Irrelevant article", "score": 0.3, "metadata": {}},
        {"content": "Another irrelevant one", "score": 0.4, "metadata": {}},
    ]

    result = await evaluate_confidence(sample_ticket_state)

    assert result["kb_confidence"] == 0.35
    assert result["action"] == "route_to_human"
    assert "low similarity" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_evaluate_high_confidence(sample_ticket_state_with_kb, mock_groq_evaluate_response):
    """Test high confidence evaluation triggers auto-respond."""
    with patch("src.agent.nodes.evaluate.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_groq_evaluate_response
        )
        mock_client.return_value = mock_instance

        result = await evaluate_confidence(sample_ticket_state_with_kb)

        assert result["kb_confidence"] == 0.9
        assert result["action"] == "auto_respond"
        assert "above threshold" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_evaluate_low_confidence(sample_ticket_state_with_kb, mock_groq_evaluate_low_response):
    """Test low confidence evaluation routes to human."""
    with patch("src.agent.nodes.evaluate.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_groq_evaluate_low_response
        )
        mock_client.return_value = mock_instance

        result = await evaluate_confidence(sample_ticket_state_with_kb)

        assert result["kb_confidence"] == 0.4
        assert result["action"] == "route_to_human"
        assert "below threshold" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_evaluate_llm_error(sample_ticket_state_with_kb):
    """Test LLM error falls back to conservative routing."""
    with patch("src.agent.nodes.evaluate.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_client.return_value = mock_instance

        result = await evaluate_confidence(sample_ticket_state_with_kb)

        assert result["kb_confidence"] == 0.3
        assert result["action"] == "route_to_human"
        assert "ERROR" in result["agent_trace"][0]
