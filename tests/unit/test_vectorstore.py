"""Unit tests for vectorstore operations."""

from unittest.mock import Mock, patch

import pytest


@pytest.mark.asyncio
async def test_vectorstore_initialization():
    """Test vectorstore initialization creates client and collection."""
    from src.rag.vectorstore import VectorStore

    with patch("chromadb.HttpClient") as mock_client:
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        vectorstore = VectorStore()
        await vectorstore.initialize()

        assert vectorstore.client is not None
        assert vectorstore.collection is not None


@pytest.mark.asyncio
async def test_vectorstore_search():
    """Test vectorstore search returns formatted results with similarity scores."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()

    # Mock search results
    vectorstore.collection.query.return_value = {
        "documents": [["Test document"]],
        "distances": [[0.2]],
        "metadatas": [[{"category": "test"}]]
    }

    results = await vectorstore.search(query="test", limit=5)

    assert len(results) == 1
    assert results[0]["content"] == "Test document"
    assert results[0]["score"] == 0.8  # 1 - 0.2
    assert results[0]["metadata"]["category"] == "test"


@pytest.mark.asyncio
async def test_vectorstore_uses_cosine_distance():
    """Test that collection is created with cosine distance metric."""
    from src.rag.vectorstore import VectorStore

    with patch("chromadb.HttpClient") as mock_client:
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        vectorstore = VectorStore()
        await vectorstore.initialize()

        # Verify cosine distance is set in collection metadata
        call_kwargs = mock_client.return_value.get_or_create_collection.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["hnsw:space"] == "cosine"


@pytest.mark.asyncio
async def test_vectorstore_uses_http_client_for_remote_host():
    """Test that HttpClient is used when CHROMA_HOST is a service name (e.g. 'chromadb')."""
    from src.rag.vectorstore import VectorStore

    with patch("chromadb.HttpClient") as mock_http, \
         patch("chromadb.PersistentClient") as mock_persistent, \
         patch("src.rag.vectorstore.settings") as mock_settings:

        mock_settings.CHROMA_HOST = "chromadb"
        mock_settings.CHROMA_PORT = 8000
        mock_settings.CHROMA_COLLECTION_NAME = "test_collection"

        mock_http.return_value.get_or_create_collection.return_value = Mock()

        vectorstore = VectorStore()
        await vectorstore.initialize()

        mock_http.assert_called_once()
        mock_persistent.assert_not_called()


@pytest.mark.asyncio
async def test_vectorstore_uses_persistent_client_for_localhost():
    """Test that PersistentClient is used for local development."""
    from src.rag.vectorstore import VectorStore

    with patch("chromadb.HttpClient") as mock_http, \
         patch("chromadb.PersistentClient") as mock_persistent, \
         patch("src.rag.vectorstore.settings") as mock_settings:

        mock_settings.CHROMA_HOST = "localhost"
        mock_settings.CHROMA_PORT = 8000
        mock_settings.CHROMA_COLLECTION_NAME = "test_collection"

        mock_persistent.return_value.get_or_create_collection.return_value = Mock()

        vectorstore = VectorStore()
        await vectorstore.initialize()

        mock_persistent.assert_called_once()
        mock_http.assert_not_called()


@pytest.mark.asyncio
async def test_vectorstore_search_with_category_filter():
    """Test search applies category filter when specified."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()
    vectorstore.collection.query.return_value = {
        "documents": [["Billing article"]],
        "distances": [[0.15]],
        "metadatas": [[{"category": "billing"}]]
    }

    await vectorstore.search(query="refund", category="billing", limit=3)

    vectorstore.collection.query.assert_called_once_with(
        query_texts=["refund"],
        n_results=3,
        where={"category": "billing"}
    )


@pytest.mark.asyncio
async def test_vectorstore_search_without_category():
    """Test search passes no filter when category is None."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()
    vectorstore.collection.query.return_value = {
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]]
    }

    await vectorstore.search(query="test")

    vectorstore.collection.query.assert_called_once_with(
        query_texts=["test"],
        n_results=5,
        where=None
    )


@pytest.mark.asyncio
async def test_vectorstore_search_empty_results():
    """Test search handles empty results gracefully."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()
    vectorstore.collection.query.return_value = {
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]]
    }

    results = await vectorstore.search(query="nonexistent")

    assert results == []


@pytest.mark.asyncio
async def test_vectorstore_search_error_returns_empty():
    """Test search returns empty list on error."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()
    vectorstore.collection.query.side_effect = Exception("Connection failed")

    results = await vectorstore.search(query="test")

    assert results == []


@pytest.mark.asyncio
async def test_vectorstore_search_not_initialized():
    """Test search raises RuntimeError when not initialized."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()

    with pytest.raises(RuntimeError, match="VectorStore not initialized"):
        await vectorstore.search(query="test")


@pytest.mark.asyncio
async def test_vectorstore_add_documents():
    """Test adding documents to the collection."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()

    docs = ["doc1", "doc2"]
    metas = [{"category": "billing"}, {"category": "technical"}]

    await vectorstore.add_documents(documents=docs, metadatas=metas, ids=["id1", "id2"])

    vectorstore.collection.add.assert_called_once_with(
        documents=docs,
        metadatas=metas,
        ids=["id1", "id2"]
    )


@pytest.mark.asyncio
async def test_vectorstore_add_documents_auto_ids():
    """Test that IDs are auto-generated when not provided."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()

    docs = ["doc1", "doc2"]
    metas = [{"category": "billing"}, {"category": "technical"}]

    await vectorstore.add_documents(documents=docs, metadatas=metas)

    call_kwargs = vectorstore.collection.add.call_args
    assert call_kwargs.kwargs["ids"] == ["doc_0", "doc_1"]


@pytest.mark.asyncio
async def test_vectorstore_score_conversion_cosine():
    """Test that cosine distances are properly converted to similarity scores."""
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore()
    vectorstore.collection = Mock()

    # With cosine distance: 0 = identical, 2 = opposite
    vectorstore.collection.query.return_value = {
        "documents": [["High match", "Medium match", "Low match"]],
        "distances": [[0.1, 0.5, 0.9]],
        "metadatas": [[{"category": "a"}, {"category": "b"}, {"category": "c"}]]
    }

    results = await vectorstore.search(query="test")

    assert results[0]["score"] == pytest.approx(0.9)   # 1 - 0.1
    assert results[1]["score"] == pytest.approx(0.5)   # 1 - 0.5
    assert results[2]["score"] == pytest.approx(0.1)   # 1 - 0.9
