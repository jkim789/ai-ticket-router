"""Pydantic request models for API endpoints."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProcessTicketRequest(BaseModel):
    """Request model for processing a support ticket."""

    message: str = Field(
        ...,
        description="Customer's support message",
        min_length=1,
        max_length=5000,
    )
    customer_id: Optional[str] = Field(
        None,
        description="Optional customer identifier",
    )
    channel: Optional[Literal["web", "email", "whatsapp", "chat"]] = Field(
        None,
        description="Channel the message came from",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "I was charged twice for my order #12345 and nobody is responding to my emails",
                    "customer_id": "cust_abc123",
                    "channel": "email",
                },
                {
                    "message": "How do I reset my password?",
                    "customer_id": "cust_def456",
                    "channel": "web",
                },
            ]
        }
    }


class SearchKBRequest(BaseModel):
    """Request model for knowledge base search."""

    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
    )
    category: Optional[str] = Field(
        None,
        description="Optional category filter",
    )
    limit: int = Field(
        5,
        description="Maximum results to return",
        ge=1,
        le=20,
    )
