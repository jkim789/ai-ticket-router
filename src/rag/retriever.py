"""
High-level retrieval pipeline for RAG.

Orchestrates query embedding, similarity search, and result formatting.
"""

import logging
from typing import Any, Dict, List, Optional

from src.rag.embeddings import EmbeddingGenerator
from src.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    High-level retrieval pipeline.

    Combines embedding generation and vector search into a simple interface.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize the retriever.

        Args:
            vectorstore: ChromaDB vector store instance
            embedding_generator: Optional embedding generator (creates one if not provided)
        """
        self.vectorstore = vectorstore
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query text
            category: Optional category filter
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0 - 1.0)

        Returns:
            List of retrieved documents with scores and metadata
        """
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")

        try:
            # Search vector store
            results = await self.vectorstore.search(
                query=query,
                category=category,
                limit=limit
            )

            # Filter by minimum score
            if min_score > 0.0:
                results = [r for r in results if r["score"] >= min_score]

            logger.info(f"Retrieved {len(results)} documents (min_score={min_score})")

            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

    async def retrieve_with_reranking(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
        rerank_limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with optional reranking.

        First retrieves more results than needed, then reranks them.
        This is a placeholder for future reranking implementation.

        Args:
            query: Search query text
            category: Optional category filter
            limit: Final number of results to return
            rerank_limit: Number of candidates to retrieve before reranking

        Returns:
            List of retrieved and reranked documents
        """
        results = await self.retrieve(
            query=query,
            category=category,
            limit=rerank_limit
        )

        # Return top N
        return results[:limit]

    async def batch_retrieve(
        self,
        queries: List[str],
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of search queries
            category: Optional category filter
            limit: Maximum results per query

        Returns:
            List of result lists (one per query)
        """
        results = []
        for query in queries:
            query_results = await self.retrieve(
                query=query,
                category=category,
                limit=limit
            )
            results.append(query_results)

        return results
