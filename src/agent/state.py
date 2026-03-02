"""
Agent state definition using TypedDict for LangGraph.

This defines the complete state structure that flows through the agent graph.
"""

from datetime import datetime
from typing import Literal, Optional, TypedDict


class TicketState(TypedDict, total=False):
    """
    Complete state for the ticket routing agent.

    This state is passed through all nodes in the LangGraph workflow.
    """

    # Input fields
    raw_message: str
    customer_id: Optional[str]
    channel: Optional[Literal["web", "email", "whatsapp", "chat"]]
    timestamp: datetime

    # Classification results
    intent: Optional[Literal["billing", "technical", "shipping", "general", "complaint", "refund"]]
    sentiment: Optional[Literal["positive", "neutral", "negative", "angry"]]
    urgency: Optional[Literal["low", "medium", "high", "critical"]]

    # RAG results
    kb_results: Optional[list[dict]]  # List of {content, score, metadata}
    kb_confidence: Optional[float]  # 0.0 - 1.0

    # Output fields
    action: Optional[Literal["auto_respond", "route_to_human"]]
    auto_response: Optional[str]
    routing: Optional[dict]  # {team, priority, summary, context}

    # Metadata
    processing_time_ms: Optional[float]
    model_used: Optional[str]
    agent_trace: Optional[list[str]]  # Step-by-step execution trace
