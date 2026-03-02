"""Knowledge base management endpoints."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import get_vectorstore
from src.api.models.responses import KBStatsResponse
from src.rag.chunker import DocumentChunker
from src.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/knowledge-base", tags=["knowledge_base"])


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    vectorstore: VectorStore = Depends(get_vectorstore)
) -> dict:
    """
    Upload and ingest a document into the knowledge base.

    Supports markdown (.md), plain text (.txt), and PDF (.pdf) files.
    Documents are automatically chunked and embedded.

    Args:
        file: Document file to ingest
        category: Category for the document (billing, technical, shipping, etc.)
        vectorstore: Injected vectorstore instance

    Returns:
        Ingestion result with document ID and chunk count

    Raises:
        HTTPException: If ingestion fails
    """
    logger.info(f"Ingesting document: {file.filename}, category: {category}")

    # Validate category
    valid_categories = ["billing", "technical", "shipping", "general", "refund", "complaint"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
        )

    # Validate file type
    allowed_extensions = [".md", ".txt", ".pdf"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)

        # Chunk the document
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)

        chunks = chunker.chunk_file(tmp_file_path, category=category)

        # Prepare for vectorstore
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

        # Add to vectorstore
        await vectorstore.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        # Clean up temp file
        tmp_file_path.unlink()

        logger.info(f"Successfully ingested {len(chunks)} chunks from {file.filename}")

        return {
            "status": "ingested",
            "document_id": file.filename,
            "chunks_created": len(chunks),
            "category": category
        }

    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        ) from e


@router.get("/search")
async def search_knowledge_base(
    q: str,
    category: Optional[str] = None,
    limit: int = 5,
    vectorstore: VectorStore = Depends(get_vectorstore)
) -> dict:
    """
    Direct similarity search against the knowledge base.

    Args:
        q: Search query text
        category: Optional category filter
        limit: Maximum number of results (1-20)
        vectorstore: Injected vectorstore instance

    Returns:
        Search results with relevance scores

    Raises:
        HTTPException: If search fails
    """
    if not q or len(q) < 2:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 2 characters"
        )

    if limit < 1 or limit > 20:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20"
        )

    try:
        results = await vectorstore.search(
            query=q,
            category=category,
            limit=limit
        )

        return {
            "query": q,
            "category": category,
            "results": [
                {
                    "content": r["content"],
                    "score": r["score"],
                    "metadata": r["metadata"]
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        ) from e


@router.get("/stats", response_model=KBStatsResponse)
async def get_knowledge_base_stats(
    vectorstore: VectorStore = Depends(get_vectorstore)
) -> KBStatsResponse:
    """
    Get knowledge base statistics.

    Returns:
        Statistics including total documents and category breakdown

    Raises:
        HTTPException: If stats retrieval fails
    """
    try:
        stats = await vectorstore.get_stats()
        return KBStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting KB stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        ) from e
