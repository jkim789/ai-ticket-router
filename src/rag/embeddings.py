"""
Embedding generation using OpenAI API.

Provides batched embedding generation with optional caching.
"""

import hashlib
import logging
from typing import Dict, List

from openai import AsyncOpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's embedding models.

    Supports batch processing and optional caching to avoid re-embedding
    unchanged documents.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        enable_cache: bool = True
    ):
        """
        Initialize the embedding generator.

        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to embed in a single API call
            enable_cache: Whether to cache embeddings
        """
        self.model = model
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.cache: Dict[str, List[float]] = {}
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Processes texts in batches and uses cache when enabled.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to log progress

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            Exception: If embedding generation fails
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            if show_progress:
                logger.info(f"Embedding batch {i // self.batch_size + 1} ({len(batch)} texts)")

            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.

        Checks cache first if enabled, then calls the API.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        results = []

        # Separate cached and uncached texts
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.enable_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    results.append(self.cache[cache_key])
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=uncached_texts
                )

                # Extract embeddings and cache them
                for i, embedding_obj in enumerate(response.data):
                    embedding = embedding_obj.embedding
                    text = uncached_texts[i]

                    if self.enable_cache:
                        cache_key = self._get_cache_key(text)
                        self.cache[cache_key] = embedding

                    results.append(embedding)

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}", exc_info=True)
                raise

        # Reorder results to match input order
        if uncached_indices:
            # Create properly ordered results
            ordered_results = [None] * len(texts)
            uncached_idx = 0

            for i in range(len(texts)):
                if i in uncached_indices:
                    ordered_results[i] = results[uncached_idx]
                    uncached_idx += 1
                else:
                    # This was cached, find it in results
                    text = texts[i]
                    cache_key = self._get_cache_key(text)
                    ordered_results[i] = self.cache[cache_key]

            return ordered_results

        return results

    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text.

        Uses MD5 hash of the text combined with the model name.

        Args:
            text: Text to generate key for

        Returns:
            Cache key string
        """
        content = f"{self.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size
        """
        return {
            "cached_embeddings": len(self.cache)
        }
