"""SQLAlchemy models for ticket history and knowledge base documents."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Ticket(Base):
    """Persisted ticket processing result."""

    __tablename__ = "tickets"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id: str = Column(String(50), unique=True, nullable=False, index=True)
    raw_message: str = Column(Text, nullable=False)
    customer_id: str | None = Column(String(100), nullable=True)
    channel: str | None = Column(String(20), nullable=True)
    intent: str = Column(String(20), nullable=False, index=True)
    sentiment: str = Column(String(20), nullable=False)
    urgency: str = Column(String(20), nullable=False)
    action: str = Column(String(20), nullable=False)
    auto_response: str | None = Column(Text, nullable=True)
    routing_team: str | None = Column(String(50), nullable=True)
    routing_priority: str | None = Column(String(20), nullable=True)
    routing_summary: str | None = Column(Text, nullable=True)
    kb_confidence: float = Column(Float, nullable=False, default=0.0)
    processing_time_ms: float = Column(Float, nullable=False)
    created_at: datetime = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        """String representation of Ticket."""
        return f"<Ticket(ticket_id={self.ticket_id}, intent={self.intent}, action={self.action})>"


class KBDocument(Base):
    """Record of an ingested knowledge base document."""

    __tablename__ = "kb_documents"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    filename: str = Column(String(255), nullable=False)
    category: str = Column(String(50), nullable=False, index=True)
    chunk_count: int = Column(Integer, nullable=False, default=0)
    created_at: datetime = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        """String representation of KBDocument."""
        return f"<KBDocument(filename={self.filename}, category={self.category})>"
