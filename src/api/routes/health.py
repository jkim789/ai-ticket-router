"""Health check endpoint."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_vectorstore
from src.api.models.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(vectorstore = Depends(get_vectorstore)) -> HealthResponse:
    """
    Health check endpoint.

    Verifies that the service and its dependencies are operational.

    Args:
        vectorstore: Injected vectorstore instance

    Returns:
        Health status of service and dependencies
    """
    # Check ChromaDB
    chromadb_status = "healthy"
    try:
        if vectorstore.collection:
            vectorstore.collection.count()
        else:
            chromadb_status = "not_initialized"
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        chromadb_status = "unhealthy"

    return HealthResponse(
        status="healthy" if chromadb_status == "healthy" else "degraded",
        chromadb=chromadb_status,
        timestamp=datetime.now()
    )
