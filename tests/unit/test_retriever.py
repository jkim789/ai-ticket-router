"""Unit tests for the retrieval pipeline."""

from unittest.mock import AsyncMock, patch

import pytest

from src.rag.retriever import Retriever


@pytest.fixture
def mock_retriever():
    """Create a Retriever with mocked dependencies."""
    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(return_value=[
        {"content": "Article 1", "score": 0.9, "metadata": {"category": "billing"}},
        {"content": "Article 2", "score": 0.7, "metadata": {"category": "billing"}},
        {"content": "Article 3", "score": 0.3, "metadata": {"category": "general"}},
    ])

    with patch("src.rag.retriever.EmbeddingGenerator"):
        retriever = Retriever(vectorstore=mock_vs)

    return retriever


@pytest.mark.asyncio
async def test_retrieve_basic(mock_retriever):
    """Test basic retrieval returns all results."""
    results = await mock_retriever.retrieve("billing question")

    assert len(results) == 3
    assert results[0]["score"] == 0.9


@pytest.mark.asyncio
async def test_retrieve_with_category(mock_retriever):
    """Test retrieval with category filter."""
    await mock_retriever.retrieve("billing question", category="billing")

    mock_retriever.vectorstore.search.assert_called_once_with(
        query="billing question",
        category="billing",
        limit=5,
    )


@pytest.mark.asyncio
async def test_retrieve_with_min_score(mock_retriever):
    """Test retrieval filters by minimum score."""
    results = await mock_retriever.retrieve("question", min_score=0.5)

    assert len(results) == 2
    assert all(r["score"] >= 0.5 for r in results)


@pytest.mark.asyncio
async def test_retrieve_error_handling(mock_retriever):
    """Test retrieval handles errors gracefully."""
    mock_retriever.vectorstore.search = AsyncMock(
        side_effect=Exception("DB Error")
    )

    results = await mock_retriever.retrieve("question")
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_with_reranking(mock_retriever):
    """Test retrieval with reranking returns limited results."""
    results = await mock_retriever.retrieve_with_reranking(
        "question", limit=2, rerank_limit=10
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_batch_retrieve(mock_retriever):
    """Test batch retrieval for multiple queries."""
    results = await mock_retriever.batch_retrieve(["q1", "q2"])

    assert len(results) == 2
    assert mock_retriever.vectorstore.search.call_count == 2
