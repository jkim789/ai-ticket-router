"""Database-backed ticket store for ticket history.

Provides persistence of processed tickets using SQLAlchemy async sessions
with SQLite backend.
"""

import logging
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.responses import TicketHistoryItem
from src.db.models import Ticket

logger = logging.getLogger(__name__)


class TicketStore:
    """
    Database-backed ticket storage for processed ticket history.

    Persists tickets to SQLite via SQLAlchemy async sessions.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize ticket store with a database session.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session

    async def add(
        self,
        ticket: TicketHistoryItem,
        raw_message: str = "",
        auto_response: Optional[str] = None,
        routing_summary: Optional[str] = None,
    ) -> None:
        """
        Persist a processed ticket to the database.

        Args:
            ticket: Ticket history item to store
            raw_message: Original customer message
            auto_response: Generated auto-response if applicable
            routing_summary: Routing summary if routed to human
        """
        db_ticket = Ticket(
            ticket_id=ticket.ticket_id,
            raw_message=raw_message,
            customer_id=ticket.customer_id,
            channel=ticket.channel,
            intent=ticket.intent,
            sentiment=ticket.sentiment,
            urgency=ticket.urgency,
            action=ticket.action,
            auto_response=auto_response,
            routing_team=ticket.routing_team,
            routing_priority=ticket.routing_priority,
            routing_summary=routing_summary,
            kb_confidence=ticket.kb_confidence,
            processing_time_ms=ticket.processing_time_ms,
            created_at=ticket.created_at,
        )
        self._session.add(db_ticket)
        await self._session.commit()
        logger.info("Ticket persisted: id=%s", ticket.ticket_id)

    async def get_history(
        self,
        limit: int = 20,
        offset: int = 0,
        intent: Optional[str] = None,
    ) -> Tuple[List[TicketHistoryItem], int]:
        """
        Retrieve ticket history with pagination and optional filtering.

        Args:
            limit: Maximum number of tickets to return
            offset: Number of tickets to skip
            intent: Optional intent category filter

        Returns:
            Tuple of (list of ticket history items, total matching count)
        """
        # Build base query
        query = select(Ticket).order_by(Ticket.created_at.desc())
        count_query = select(func.count()).select_from(Ticket)

        if intent:
            query = query.where(Ticket.intent == intent)
            count_query = count_query.where(Ticket.intent == intent)

        # Get total count
        total_result = await self._session.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results
        query = query.offset(offset).limit(limit)
        result = await self._session.execute(query)
        rows = result.scalars().all()

        tickets = [
            TicketHistoryItem(
                ticket_id=row.ticket_id,
                action=row.action,
                intent=row.intent,
                sentiment=row.sentiment,
                urgency=row.urgency,
                routing_team=row.routing_team,
                routing_priority=row.routing_priority,
                kb_confidence=row.kb_confidence,
                processing_time_ms=row.processing_time_ms,
                customer_id=row.customer_id,
                channel=row.channel,
                created_at=row.created_at,
            )
            for row in rows
        ]

        return tickets, total
