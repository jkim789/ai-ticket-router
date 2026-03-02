"""Tests for database models and session management."""

from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base, KBDocument, Ticket


@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


class TestTicketModel:
    """Tests for the Ticket SQLAlchemy model."""

    @pytest.mark.asyncio
    async def test_create_ticket(self, db_session):
        """Should create and retrieve a ticket."""
        ticket = Ticket(
            ticket_id="tkt_abc123",
            raw_message="Help with billing",
            customer_id="cust_1",
            channel="web",
            intent="billing",
            sentiment="neutral",
            urgency="medium",
            action="auto_respond",
            auto_response="Here is your answer.",
            kb_confidence=0.85,
            processing_time_ms=1234.5,
            created_at=datetime.now(),
        )
        db_session.add(ticket)
        await db_session.commit()

        result = await db_session.execute(
            select(Ticket).where(Ticket.ticket_id == "tkt_abc123")
        )
        saved = result.scalar_one()
        assert saved.intent == "billing"
        assert saved.raw_message == "Help with billing"
        assert saved.kb_confidence == 0.85

    @pytest.mark.asyncio
    async def test_ticket_with_routing(self, db_session):
        """Should store routing fields for human-routed tickets."""
        ticket = Ticket(
            ticket_id="tkt_routed",
            raw_message="I was charged twice!",
            intent="billing",
            sentiment="angry",
            urgency="high",
            action="route_to_human",
            routing_team="billing_team",
            routing_priority="critical",
            routing_summary="Customer reports duplicate charge.",
            kb_confidence=0.45,
            processing_time_ms=2000.0,
            created_at=datetime.now(),
        )
        db_session.add(ticket)
        await db_session.commit()

        result = await db_session.execute(
            select(Ticket).where(Ticket.ticket_id == "tkt_routed")
        )
        saved = result.scalar_one()
        assert saved.action == "route_to_human"
        assert saved.routing_team == "billing_team"
        assert saved.routing_priority == "critical"

    @pytest.mark.asyncio
    async def test_ticket_nullable_fields(self, db_session):
        """Optional fields should accept None."""
        ticket = Ticket(
            ticket_id="tkt_minimal",
            raw_message="Help",
            intent="general",
            sentiment="neutral",
            urgency="low",
            action="auto_respond",
            kb_confidence=0.9,
            processing_time_ms=500.0,
            created_at=datetime.now(),
        )
        db_session.add(ticket)
        await db_session.commit()

        result = await db_session.execute(
            select(Ticket).where(Ticket.ticket_id == "tkt_minimal")
        )
        saved = result.scalar_one()
        assert saved.customer_id is None
        assert saved.channel is None
        assert saved.routing_team is None

    @pytest.mark.asyncio
    async def test_ticket_repr(self, db_session):
        """Ticket __repr__ should include key fields."""
        ticket = Ticket(
            ticket_id="tkt_repr",
            raw_message="Test",
            intent="billing",
            sentiment="neutral",
            urgency="low",
            action="auto_respond",
            kb_confidence=0.9,
            processing_time_ms=100.0,
            created_at=datetime.now(),
        )
        assert "tkt_repr" in repr(ticket)
        assert "billing" in repr(ticket)


class TestKBDocumentModel:
    """Tests for the KBDocument SQLAlchemy model."""

    @pytest.mark.asyncio
    async def test_create_kb_document(self, db_session):
        """Should create and retrieve a KB document record."""
        doc = KBDocument(
            filename="billing.md",
            category="billing",
            chunk_count=18,
            created_at=datetime.now(),
        )
        db_session.add(doc)
        await db_session.commit()

        result = await db_session.execute(
            select(KBDocument).where(KBDocument.filename == "billing.md")
        )
        saved = result.scalar_one()
        assert saved.category == "billing"
        assert saved.chunk_count == 18

    @pytest.mark.asyncio
    async def test_kb_document_repr(self, db_session):
        """KBDocument __repr__ should include key fields."""
        doc = KBDocument(
            filename="test.md",
            category="technical",
            chunk_count=5,
            created_at=datetime.now(),
        )
        assert "test.md" in repr(doc)
        assert "technical" in repr(doc)
