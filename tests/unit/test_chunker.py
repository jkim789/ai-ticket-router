"""Unit tests for document chunker."""

from src.rag.chunker import DocumentChunker


def test_chunk_text_basic():
    """Test basic text chunking."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    text = "This is a test. " * 50  # Create text that needs chunking

    chunks = chunker.chunk_text(text, metadata={"source": "test"})

    assert len(chunks) > 1
    assert all("source" in chunk["metadata"] for chunk in chunks)
    assert all("chunk_index" in chunk["metadata"] for chunk in chunks)


def test_chunk_markdown_with_headers():
    """Test markdown chunking preserves headers."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    markdown = """# Header 1

This is content under header 1.

## Header 2

This is content under header 2.
"""

    chunks = chunker.chunk_markdown(markdown, metadata={"category": "test"})

    assert len(chunks) > 0
    # Check that some chunks have section metadata
    sections = [c["metadata"].get("section") for c in chunks if "section" in c["metadata"]]
    assert len(sections) > 0


def test_clean_text():
    """Test text cleaning."""
    chunker = DocumentChunker()

    dirty_text = "Test   with    extra   spaces\n\n\n\nand   lines"
    clean = chunker._clean_text(dirty_text)

    assert "   " not in clean
    assert "\n\n\n" not in clean


def test_chunk_size_respected():
    """Test that chunks respect size limits."""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)

    # Create long text
    text = "word " * 200

    chunks = chunker.chunk_text(text)

    # Check that chunks are roughly the right size (allowing some variance)
    for chunk in chunks:
        # Rough token estimate: 1 token ≈ 4 chars
        estimated_tokens = len(chunk["text"]) // 4
        assert estimated_tokens <= 70  # Allow some overflow
