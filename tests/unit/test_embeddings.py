"""Unit tests for embedding generation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.rag.embeddings import EmbeddingGenerator


@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator with mocked OpenAI client."""
    with patch("src.rag.embeddings.AsyncOpenAI"):
        gen = EmbeddingGenerator(model="text-embedding-3-small", enable_cache=True)
        return gen


@pytest.fixture
def mock_embedding_response():
    """Mock OpenAI embeddings response."""
    mock_data_1 = Mock()
    mock_data_1.embedding = [0.1, 0.2, 0.3]
    mock_data_2 = Mock()
    mock_data_2.embedding = [0.4, 0.5, 0.6]
    mock_response = Mock()
    mock_response.data = [mock_data_1, mock_data_2]
    return mock_response


@pytest.mark.asyncio
async def test_embed_texts_empty(embedding_generator):
    """Test embedding empty list returns empty list."""
    result = await embedding_generator.embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_texts_success(embedding_generator, mock_embedding_response):
    """Test successful embedding generation."""
    embedding_generator.client.embeddings.create = AsyncMock(
        return_value=mock_embedding_response
    )

    result = await embedding_generator.embed_texts(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_embed_text_single(embedding_generator):
    """Test single text embedding."""
    mock_data = Mock()
    mock_data.embedding = [0.1, 0.2, 0.3]
    mock_response = Mock()
    mock_response.data = [mock_data]

    embedding_generator.client.embeddings.create = AsyncMock(
        return_value=mock_response
    )

    result = await embedding_generator.embed_text("hello")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_texts_uses_cache(embedding_generator, mock_embedding_response):
    """Test that cache prevents duplicate API calls."""
    embedding_generator.client.embeddings.create = AsyncMock(
        return_value=mock_embedding_response
    )

    # First call - should hit API
    await embedding_generator.embed_texts(["hello", "world"])
    assert embedding_generator.client.embeddings.create.call_count == 1

    # Second call with same texts - should use cache
    mock_single = Mock()
    mock_single.data = []
    embedding_generator.client.embeddings.create = AsyncMock(return_value=mock_single)

    result = await embedding_generator.embed_texts(["hello", "world"])
    assert len(result) == 2
    # Should not have called API again (all cached)
    embedding_generator.client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_embed_texts_api_error(embedding_generator):
    """Test error propagation on API failure."""
    embedding_generator.client.embeddings.create = AsyncMock(
        side_effect=Exception("API Error")
    )

    with pytest.raises(Exception, match="API Error"):
        await embedding_generator.embed_texts(["hello"])


def test_cache_key_generation(embedding_generator):
    """Test cache key is deterministic."""
    key1 = embedding_generator._get_cache_key("hello")
    key2 = embedding_generator._get_cache_key("hello")
    key3 = embedding_generator._get_cache_key("world")

    assert key1 == key2
    assert key1 != key3


def test_clear_cache(embedding_generator):
    """Test cache clearing."""
    embedding_generator.cache["test"] = [0.1]
    assert len(embedding_generator.cache) == 1

    embedding_generator.clear_cache()
    assert len(embedding_generator.cache) == 0


def test_get_cache_stats(embedding_generator):
    """Test cache statistics."""
    assert embedding_generator.get_cache_stats() == {"cached_embeddings": 0}

    embedding_generator.cache["test"] = [0.1]
    assert embedding_generator.get_cache_stats() == {"cached_embeddings": 1}
