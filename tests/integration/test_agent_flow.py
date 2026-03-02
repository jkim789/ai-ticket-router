"""Integration tests for the LangGraph agent flow."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Set required env vars before any app imports
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Mock chromadb to avoid import issues
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.config", MagicMock())

from datetime import datetime

import pytest

from src.agent.graph import build_graph, should_auto_respond


class TestShouldAutoRespond:
    """Tests for the conditional routing function."""

    def test_auto_respond_action(self):
        """Test routing to generate_response when action is auto_respond."""
        state = {"action": "auto_respond"}
        assert should_auto_respond(state) == "generate_response"

    def test_route_to_human_action(self):
        """Test routing to route_to_human when action is route_to_human."""
        state = {"action": "route_to_human"}
        assert should_auto_respond(state) == "route_to_human"

    def test_missing_action_defaults_to_human(self):
        """Test default routing when action is missing."""
        state = {}
        assert should_auto_respond(state) == "route_to_human"

    def test_unknown_action_defaults_to_human(self):
        """Test unknown action value routes to human."""
        state = {"action": "unknown_action"}
        assert should_auto_respond(state) == "route_to_human"


class TestBuildGraph:
    """Tests for graph construction."""

    def test_build_graph_compiles(self):
        """Test that build_graph returns a compiled graph."""
        mock_vectorstore = AsyncMock()
        graph = build_graph(mock_vectorstore)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_auto_respond(self):
        """Test full pipeline resulting in auto-response."""
        mock_vectorstore = AsyncMock()

        # Build the graph
        graph = build_graph(mock_vectorstore)

        # Mock all LLM calls
        classify_response = Mock()
        classify_choice = Mock()
        classify_choice.message.content = '{"intent": "billing", "sentiment": "neutral", "urgency": "medium", "reasoning": "Test"}'
        classify_response.choices = [classify_choice]

        evaluate_response = Mock()
        evaluate_choice = Mock()
        evaluate_choice.message.content = '{"can_answer": true, "confidence": 0.9, "reasoning": "Good match"}'
        evaluate_response.choices = [evaluate_choice]

        respond_response = Mock()
        respond_choice = Mock()
        respond_choice.message.content = "Here is how to fix your billing issue."
        respond_response.choices = [respond_choice]

        # Set up vectorstore to return results
        mock_vectorstore.search = AsyncMock(return_value=[
            {"content": "Billing FAQ", "score": 0.85, "metadata": {"category": "billing"}}
        ])

        # Sequence LLM responses: classify, evaluate, respond
        call_count = 0
        responses = [classify_response, evaluate_response, respond_response]

        async def mock_create(**kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch("src.agent.nodes.classify.get_llm_client") as mock_classify, \
             patch("src.agent.nodes.evaluate.get_llm_client") as mock_evaluate, \
             patch("src.agent.nodes.respond.get_llm_client") as mock_respond:

            for mock_client, response in [
                (mock_classify, classify_response),
                (mock_evaluate, evaluate_response),
                (mock_respond, respond_response),
            ]:
                instance = AsyncMock()
                instance.chat.completions.create = AsyncMock(return_value=response)
                mock_client.return_value = instance

            state = {
                "raw_message": "Help with billing",
                "customer_id": "test_123",
                "channel": "email",
                "timestamp": datetime.now(),
                "agent_trace": [],
            }

            result = await graph.ainvoke(state)

            assert result["intent"] == "billing"
            assert result["action"] == "auto_respond"
            assert result["auto_response"] is not None
            assert len(result["agent_trace"]) >= 3

    @pytest.mark.asyncio
    async def test_full_pipeline_route_to_human(self):
        """Test full pipeline resulting in human routing."""
        mock_vectorstore = AsyncMock()
        mock_vectorstore.search = AsyncMock(return_value=[])

        graph = build_graph(mock_vectorstore)

        classify_response = Mock()
        classify_choice = Mock()
        classify_choice.message.content = '{"intent": "complaint", "sentiment": "angry", "urgency": "high", "reasoning": "Angry customer"}'
        classify_response.choices = [classify_choice]

        route_response = Mock()
        route_choice = Mock()
        route_choice.message.content = "Customer is very upset about service quality."
        route_response.choices = [route_choice]

        with patch("src.agent.nodes.classify.get_llm_client") as mock_classify, \
             patch("src.agent.nodes.route.get_llm_client") as mock_route:

            classify_instance = AsyncMock()
            classify_instance.chat.completions.create = AsyncMock(return_value=classify_response)
            mock_classify.return_value = classify_instance

            route_instance = AsyncMock()
            route_instance.chat.completions.create = AsyncMock(return_value=route_response)
            mock_route.return_value = route_instance

            state = {
                "raw_message": "This is unacceptable! Your service is terrible!",
                "customer_id": "test_456",
                "channel": "web",
                "timestamp": datetime.now(),
                "agent_trace": [],
            }

            result = await graph.ainvoke(state)

            assert result["intent"] == "complaint"
            assert result["action"] == "route_to_human"
            assert result["routing"]["team"] == "escalation"
            assert result["routing"]["priority"] == "critical"
