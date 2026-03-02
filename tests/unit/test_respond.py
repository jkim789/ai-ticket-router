"""Unit tests for auto-response generation node."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.respond import generate_response


@pytest.mark.asyncio
async def test_generate_response_success(sample_ticket_state_with_kb, mock_groq_respond_response):
    """Test successful response generation."""
    with patch("src.agent.nodes.respond.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_groq_respond_response
        )
        mock_client.return_value = mock_instance

        result = await generate_response(sample_ticket_state_with_kb)

        assert result["auto_response"] is not None
        assert len(result["auto_response"]) > 0
        assert "generate_response" in result["agent_trace"][0]
        assert "char response" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_generate_response_fallback_on_error(sample_ticket_state_with_kb):
    """Test fallback response when LLM fails."""
    with patch("src.agent.nodes.respond.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_client.return_value = mock_instance

        result = await generate_response(sample_ticket_state_with_kb)

        assert "novatech" in result["auto_response"].lower()
        assert "support team" in result["auto_response"].lower()
        assert "ERROR" in result["agent_trace"][0]


@pytest.mark.asyncio
async def test_generate_response_empty_kb(sample_ticket_state, mock_groq_respond_response):
    """Test response generation with no KB results."""
    sample_ticket_state["kb_results"] = []

    with patch("src.agent.nodes.respond.get_llm_client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_groq_respond_response
        )
        mock_client.return_value = mock_instance

        result = await generate_response(sample_ticket_state)

        assert result["auto_response"] is not None
        mock_instance.chat.completions.create.assert_called_once()
