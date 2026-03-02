"""FastAPI dependency injection."""

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.ticket_store import TicketStore
from src.db.database import get_session


def get_vectorstore(request: Request) -> Any:
    """
    Dependency to inject vectorstore instance.

    Args:
        request: FastAPI request object

    Returns:
        VectorStore instance from app state
    """
    return request.app.state.vectorstore


def get_agent_graph(request: Request) -> Any:
    """
    Dependency to inject agent graph instance.

    Args:
        request: FastAPI request object

    Returns:
        Compiled LangGraph instance from app state
    """
    return request.app.state.agent_graph


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to inject an async database session.

    Yields:
        AsyncSession for database operations
    """
    async for session in get_session():
        yield session


async def get_ticket_store(
    session: AsyncSession = Depends(get_db_session),
) -> TicketStore:
    """
    Dependency to inject a database-backed ticket store.

    Args:
        session: Injected async database session

    Returns:
        TicketStore instance backed by the database
    """
    return TicketStore(session)
