"""Ticket processing and history endpoints."""

import logging
import time
import uuid
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.agent.state import TicketState
from src.api.dependencies import get_agent_graph, get_ticket_store
from src.api.models.requests import ProcessTicketRequest
from src.api.models.responses import (
    ClassificationResponse,
    ErrorResponse,
    ProcessTicketResponse,
    RoutingResponse,
    TicketHistoryItem,
    TicketHistoryResponse,
)
from src.api.ticket_store import TicketStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


@router.post(
    "/process",
    response_model=ProcessTicketResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Processing failed"},
    },
    summary="Process a support ticket",
    description=(
        "Runs a customer support message through the full AI agent pipeline: "
        "intent classification, knowledge base search, confidence evaluation, "
        "and either auto-response generation or human routing."
    ),
)
async def process_ticket(
    request: ProcessTicketRequest,
    graph=Depends(get_agent_graph),
    ticket_store: TicketStore = Depends(get_ticket_store),
) -> ProcessTicketResponse:
    """
    Process a customer support ticket through the agent pipeline.

    This endpoint runs the complete LangGraph workflow:
    1. Classifies intent, sentiment, and urgency
    2. Searches knowledge base for relevant articles
    3. Evaluates confidence in KB results
    4. Either generates auto-response or creates human handoff packet

    Args:
        request: Ticket processing request with message and optional metadata
        graph: Injected LangGraph instance
        ticket_store: Injected database-backed ticket store

    Returns:
        Complete processing results including action taken and agent trace

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()

    try:
        # Build initial state
        initial_state: TicketState = {
            "raw_message": request.message,
            "customer_id": request.customer_id,
            "channel": request.channel,
            "timestamp": datetime.now(),
            "agent_trace": [],
        }

        # Run the graph
        logger.info("Processing ticket for customer: %s", request.customer_id)
        result = await graph.ainvoke(initial_state)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time_ms

        # Format response
        classification = ClassificationResponse(
            intent=result["intent"],
            sentiment=result["sentiment"],
            urgency=result["urgency"],
        )

        routing = None
        routing_summary = None
        if result["action"] == "route_to_human" and "routing" in result:
            routing_data = result["routing"]
            routing = RoutingResponse(
                team=routing_data["team"],
                priority=routing_data["priority"],
                summary=routing_data["summary"],
                relevant_kb_articles=routing_data.get("context", []),
            )
            routing_summary = routing_data["summary"]

        ticket_id = f"tkt_{uuid.uuid4().hex[:12]}"

        response = ProcessTicketResponse(
            ticket_id=ticket_id,
            action=result["action"],
            classification=classification,
            routing=routing,
            auto_response=result.get("auto_response"),
            kb_confidence=result.get("kb_confidence", 0.0),
            processing_time_ms=processing_time_ms,
            agent_trace=result.get("agent_trace", []),
        )

        # Persist ticket to database
        history_item = TicketHistoryItem(
            ticket_id=ticket_id,
            action=result["action"],
            intent=result["intent"],
            sentiment=result["sentiment"],
            urgency=result["urgency"],
            routing_team=routing.team if routing else None,
            routing_priority=routing.priority if routing else None,
            kb_confidence=result.get("kb_confidence", 0.0),
            processing_time_ms=processing_time_ms,
            customer_id=request.customer_id,
            channel=request.channel,
            created_at=datetime.now(),
        )
        await ticket_store.add(
            ticket=history_item,
            raw_message=request.message,
            auto_response=result.get("auto_response"),
            routing_summary=routing_summary,
        )

        logger.info(
            "Ticket processed: id=%s action=%s time=%.0fms",
            ticket_id,
            result["action"],
            processing_time_ms,
        )

        return response

    except Exception as e:
        logger.error("Error processing ticket: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process ticket: {str(e)}",
        ) from e


@router.get(
    "/history",
    response_model=TicketHistoryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
    },
    summary="Get ticket processing history",
    description=(
        "Returns paginated history of processed tickets from the database. "
        "Supports filtering by intent category."
    ),
)
async def get_ticket_history(
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="Maximum number of tickets to return",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of tickets to skip",
    ),
    intent: Optional[
        Literal["billing", "technical", "shipping", "general", "complaint", "refund"]
    ] = Query(None, description="Filter by intent category"),
    ticket_store: TicketStore = Depends(get_ticket_store),
) -> TicketHistoryResponse:
    """
    Return processed ticket history with pagination and optional filtering.

    Queries the SQLite database for persisted tickets.

    Args:
        limit: Maximum number of results (1-100, default 20)
        offset: Number of results to skip (default 0)
        intent: Optional intent category filter
        ticket_store: Injected database-backed ticket store

    Returns:
        Paginated list of processed tickets
    """
    tickets, total = await ticket_store.get_history(
        limit=limit, offset=offset, intent=intent
    )

    return TicketHistoryResponse(
        tickets=tickets,
        total=total,
        limit=limit,
        offset=offset,
    )
