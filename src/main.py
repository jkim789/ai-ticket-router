"""
FastAPI application entry point.

This module initializes the FastAPI app, sets up middleware, includes routes,
and manages application lifecycle (startup/shutdown).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.agent.graph import build_graph
from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.routes import health, knowledge_base, tickets
from src.config import settings
from src.db.database import close_db, init_db
from src.rag.vectorstore import VectorStore

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown operations:
    - Startup: Initialize database, ChromaDB, and compile LangGraph
    - Shutdown: Close connections gracefully
    """
    logger.info("Starting AI Ticket Router...")

    # Initialize database
    await init_db()

    # Initialize vector store
    vectorstore = VectorStore()
    await vectorstore.initialize()
    app.state.vectorstore = vectorstore

    # Build and compile agent graph
    agent_graph = build_graph(vectorstore)
    app.state.agent_graph = agent_graph

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    await vectorstore.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AI Ticket Router",
    description=(
        "AI-powered customer support ticket routing agent. "
        "Classifies incoming messages, searches a knowledge base via RAG, "
        "and either auto-responds or routes to the appropriate human team."
    ),
    version="0.3.0",
    lifespan=lifespan,
    responses={
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "internal_error",
                        "detail": "An unexpected error occurred",
                        "path": "/api/v1/tickets/process",
                    }
                }
            },
        }
    },
)

# Add middleware (order: rate limit runs before logging)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(tickets.router)
app.include_router(knowledge_base.router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the demo frontend.

    Returns:
        HTML response with the demo UI
    """
    frontend_path = Path(__file__).parent / "frontend" / "index.html"

    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text())

    return HTMLResponse(
        content="""
        <html>
            <head><title>AI Ticket Router</title></head>
            <body>
                <h1>AI Ticket Router API</h1>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """
    )
