"""
ChromaDB vector store operations.

Handles document storage, retrieval, and collection management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB vector store client.

    Provides high-level interface for document storage and semantic search.
    """

    def __init__(self):
        """Initialize ChromaDB client."""
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection = None

    async def initialize(self) -> None:
        """
        Initialize connection to ChromaDB and get/create collection.

        Uses persistent local storage in development mode and HTTP client
        when a remote ChromaDB host is configured (e.g. Docker).

        Raises:
            Exception: If connection or collection creation fails
        """
        try:
            if settings.CHROMA_HOST in ("localhost", "127.0.0.1"):
                # Local/dev mode: use persistent embedded client
                chroma_path = str(Path(__file__).parent.parent.parent / "data" / "chroma")
                self.client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=ChromaSettings(
                        anonymized_telemetry=False
                    ),
                )
                logger.info("ChromaDB using persistent local storage: %s", chroma_path)
            else:
                self.client = chromadb.HttpClient(
                    host=settings.CHROMA_HOST,
                    port=settings.CHROMA_PORT,
                    settings=ChromaSettings(
                        anonymized_telemetry=False
                    ),
                )

            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "NovaTech Solutions knowledge base",
                    "hnsw:space": "cosine"
                }
            )

            logger.info(f"ChromaDB initialized: collection={settings.CHROMA_COLLECTION_NAME}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search in the knowledge base.

        Args:
            query: Search query text
            category: Optional category filter (intent type)
            limit: Maximum number of results

        Returns:
            List of results with content, score, and metadata

        Raises:
            RuntimeError: If ChromaDB not initialized
        """
        if not self.collection:
            raise RuntimeError("VectorStore not initialized")

        try:
            # Build query filters
            where_filter = {"category": category} if category else None

            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter
            )

            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })

            logger.info(f"Search completed: {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts (must include 'category')
            ids: Optional list of document IDs

        Raises:
            RuntimeError: If ChromaDB not initialized
        """
        if not self.collection:
            raise RuntimeError("VectorStore not initialized")

        try:
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} documents to knowledge base")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dict with total_documents, total_chunks, and category breakdown
        """
        if not self.collection:
            raise RuntimeError("VectorStore not initialized")

        try:
            count = self.collection.count()

            # Get category breakdown (simplified - would need full scan for accuracy)
            return {
                "total_documents": count,
                "total_chunks": count,
                "categories": {}
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {"total_documents": 0, "total_chunks": 0, "categories": {}}

    async def close(self) -> None:
        """Close ChromaDB connection."""
        logger.info("ChromaDB connection closed")
