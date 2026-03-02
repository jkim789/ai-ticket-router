"""Async SQLite database session management."""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory — initialized via init_db()
_engine = None
_session_factory = None


async def init_db() -> None:
    """
    Initialize the async database engine and create tables.

    Creates the SQLite database file and all tables defined in models.py
    if they don't already exist.
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.APP_ENV == "development",
    )
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized: %s", settings.DATABASE_URL)


async def close_db() -> None:
    """Close the database engine and release connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connection closed")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an async database session.

    Yields:
        AsyncSession bound to the current engine

    Raises:
        RuntimeError: If the database has not been initialized
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _session_factory() as session:
        yield session
