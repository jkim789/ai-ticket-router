"""Tests for database-backed ticket store."""

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.models.responses import TicketHistoryItem
from src.api.ticket_store import TicketStore
from src.db.models import Base


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


def _make_ticket(
    ticket_id: str = "tkt_test123",
    intent: str = "billing",
    action: str = "auto_respond",
) -> TicketHistoryItem:
    """Helper to create a test ticket history item."""
    return TicketHistoryItem(
        ticket_id=ticket_id,
        action=action,
        intent=intent,
        sentiment="neutral",
        urgency="medium",
        kb_confidence=0.85,
        processing_time_ms=1234.5,
        customer_id="cust_test",
        channel="web",
        created_at=datetime.now(),
    )


class TestTicketStore:
    """Tests for TicketStore with database backend."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self, db_session):
        """Adding a ticket should make it retrievable."""
        store = TicketStore(db_session)
        await store.add(_make_ticket(), raw_message="Help with billing")

        tickets, total = await store.get_history()
        assert total == 1
        assert tickets[0].ticket_id == "tkt_test123"

    @pytest.mark.asyncio
    async def test_newest_first_ordering(self, db_session):
        """Tickets should be returned newest first."""
        store = TicketStore(db_session)
        await store.add(
            _make_ticket(ticket_id="tkt_first"),
            raw_message="First message",
        )
        await store.add(
            _make_ticket(ticket_id="tkt_second"),
            raw_message="Second message",
        )

        tickets, _ = await store.get_history()
        assert tickets[0].ticket_id == "tkt_second"
        assert tickets[1].ticket_id == "tkt_first"

    @pytest.mark.asyncio
    async def test_pagination_limit(self, db_session):
        """Limit should restrict the number of returned tickets."""
        store = TicketStore(db_session)
        for i in range(5):
            await store.add(
                _make_ticket(ticket_id=f"tkt_{i}"),
                raw_message=f"Message {i}",
            )

        tickets, total = await store.get_history(limit=2)
        assert len(tickets) == 2
        assert total == 5

    @pytest.mark.asyncio
    async def test_pagination_offset(self, db_session):
        """Offset should skip the specified number of tickets."""
        store = TicketStore(db_session)
        for i in range(5):
            await store.add(
                _make_ticket(ticket_id=f"tkt_{i}"),
                raw_message=f"Message {i}",
            )

        tickets, total = await store.get_history(limit=2, offset=3)
        assert len(tickets) == 2
        assert total == 5

    @pytest.mark.asyncio
    async def test_filter_by_intent(self, db_session):
        """Filtering by intent should return only matching tickets."""
        store = TicketStore(db_session)
        await store.add(
            _make_ticket(ticket_id="tkt_billing", intent="billing"),
            raw_message="Billing issue",
        )
        await store.add(
            _make_ticket(ticket_id="tkt_tech", intent="technical"),
            raw_message="Tech issue",
        )
        await store.add(
            _make_ticket(ticket_id="tkt_billing2", intent="billing"),
            raw_message="Another billing issue",
        )

        tickets, total = await store.get_history(intent="billing")
        assert total == 2
        assert all(t.intent == "billing" for t in tickets)

    @pytest.mark.asyncio
    async def test_filter_with_pagination(self, db_session):
        """Filtering and pagination should work together."""
        store = TicketStore(db_session)
        for i in range(5):
            await store.add(
                _make_ticket(ticket_id=f"tkt_b{i}", intent="billing"),
                raw_message=f"Billing {i}",
            )
        await store.add(
            _make_ticket(ticket_id="tkt_tech", intent="technical"),
            raw_message="Tech issue",
        )

        tickets, total = await store.get_history(limit=2, offset=1, intent="billing")
        assert len(tickets) == 2
        assert total == 5

    @pytest.mark.asyncio
    async def test_empty_store(self, db_session):
        """Empty store should return empty list with zero total."""
        store = TicketStore(db_session)
        tickets, total = await store.get_history()
        assert tickets == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_filter_no_matches(self, db_session):
        """Filtering with no matches should return empty list."""
        store = TicketStore(db_session)
        await store.add(
            _make_ticket(intent="billing"),
            raw_message="Billing issue",
        )

        tickets, total = await store.get_history(intent="shipping")
        assert tickets == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_persists_extra_fields(self, db_session):
        """Auto-response and routing summary should be persisted."""
        store = TicketStore(db_session)
        await store.add(
            _make_ticket(),
            raw_message="Help me",
            auto_response="Here is your answer.",
            routing_summary="Route to billing team.",
        )

        tickets, total = await store.get_history()
        assert total == 1
