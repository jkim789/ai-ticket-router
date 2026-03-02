"""Unit tests for intent classification node."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.classify import classify_intent


@pytest.mark.asyncio
async def test_classify_intent_success(sample_ticket_state, mock_groq_response):
    """Test successful intent classification."""
    with patch("src.agent.nodes.classify.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_groq_response
        )
        mock_client.return_value = mock_instance

        result = await classify_intent(sample_ticket_state)

        assert result["intent"] == "billing"
        assert result["sentiment"] in ["positive", "neutral", "negative", "angry"]
        assert result["urgency"] in ["low", "medium", "high", "critical"]
        assert len(result["agent_trace"]) > 0
        assert "classify_intent" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_classify_intent_fallback_on_error(sample_ticket_state):
    """Test fallback behavior when classification fails."""
    with patch("src.agent.nodes.classify.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_client.return_value = mock_instance

        result = await classify_intent(sample_ticket_state)

        # Should fallback to general with neutral sentiment
        assert result["intent"] == "general"
        assert result["sentiment"] == "neutral"
        assert result["urgency"] == "medium"
        assert "ERROR" in result["agent_trace"][0]
