"""Pydantic response models for API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Human-readable error message")
    path: Optional[str] = Field(None, description="Request path that caused the error")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "validation_error",
                    "detail": "Message must be at least 1 character",
                    "path": "/api/v1/tickets/process",
                }
            ]
        }
    }


class ClassificationResponse(BaseModel):
    """Classification results."""

    intent: Literal["billing", "technical", "shipping", "general", "complaint", "refund"]
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    urgency: Literal["low", "medium", "high", "critical"]


class RoutingResponse(BaseModel):
    """Routing information for human handoff."""

    team: str
    priority: str
    summary: str
    relevant_kb_articles: List[Dict[str, Any]] = Field(default_factory=list)


class ProcessTicketResponse(BaseModel):
    """Response model for processed ticket."""

    ticket_id: str
    action: Literal["auto_respond", "route_to_human"]
    classification: ClassificationResponse
    routing: Optional[RoutingResponse] = None
    auto_response: Optional[str] = None
    kb_confidence: float
    processing_time_ms: float
    agent_trace: List[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticket_id": "tkt_abc123",
                    "action": "route_to_human",
                    "classification": {
                        "intent": "billing",
                        "sentiment": "angry",
                        "urgency": "high",
                    },
                    "routing": {
                        "team": "billing_team",
                        "priority": "high",
                        "summary": "Customer reports duplicate charge on order #12345.",
                        "relevant_kb_articles": [
                            {"title": "Billing Disputes", "relevance_score": 0.82}
                        ],
                    },
                    "kb_confidence": 0.68,
                    "processing_time_ms": 2340.5,
                    "agent_trace": [
                        "classify_intent: billing (confidence: 0.95)",
                        "search_kb: found 5 results, top score 0.82",
                        "evaluate_confidence: 0.68 (below threshold)",
                        "route_to_human: billing_team, priority high",
                    ],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    chromadb: str
    timestamp: datetime


class KBStatsResponse(BaseModel):
    """Knowledge base statistics response."""

    total_documents: int
    total_chunks: int
    categories: Dict[str, int]


class TicketHistoryItem(BaseModel):
    """Single ticket in history list."""

    ticket_id: str
    action: Literal["auto_respond", "route_to_human"]
    intent: str
    sentiment: str
    urgency: str
    routing_team: Optional[str] = None
    routing_priority: Optional[str] = None
    kb_confidence: float
    processing_time_ms: float
    customer_id: Optional[str] = None
    channel: Optional[str] = None
    created_at: datetime


class TicketHistoryResponse(BaseModel):
    """Paginated ticket history response."""

    tickets: List[TicketHistoryItem]
    total: int
    limit: int
    offset: int
