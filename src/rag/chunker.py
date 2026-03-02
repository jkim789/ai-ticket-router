"""
Document chunking strategies for RAG pipeline.

Implements recursive text splitting with support for multiple document formats.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Document chunker with support for markdown, plain text, and PDF.

    Uses recursive text splitting to maintain semantic coherence while
    respecting token limits.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            separators: List of separator strings for recursive splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators for recursive splitting
        self.separators = separators or [
            "\n\n\n",  # Multiple blank lines
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            "! ",      # Sentences
            "? ",      # Sentences
            "; ",      # Clauses
            ", ",      # Phrases
            " ",       # Words
            ""         # Characters
        ]

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with metadata preservation.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        # Clean the text
        text = self._clean_text(text)

        # Perform recursive splitting
        chunks = self._recursive_split(text, self.separators)

        # Format chunks with metadata
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            formatted_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

        logger.info(f"Created {len(formatted_chunks)} chunks from text")
        return formatted_chunks

    def chunk_markdown(
        self,
        markdown_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk markdown text while preserving structure.

        Extracts headers and includes them in chunk metadata for better context.

        Args:
            markdown_text: Markdown formatted text
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Extract sections based on headers
        sections = self._split_by_headers(markdown_text)

        all_chunks = []
        for section in sections:
            header = section.get("header", "")
            content = section.get("content", "")

            # Create metadata with header context
            section_metadata = metadata.copy() if metadata else {}
            if header:
                section_metadata["section"] = header

            # Chunk the section content
            chunks = self.chunk_text(content, section_metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_file(
        self,
        file_path: Path,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a file based on its extension.

        Args:
            file_path: Path to the file
            category: Optional category for the document

        Returns:
            List of chunk dictionaries

        Raises:
            ValueError: If file type is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Base metadata
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
        }
        if category:
            metadata["category"] = category

        # Read file content
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            content = file_path.read_text(encoding="utf-8")
            return self.chunk_markdown(content, metadata)

        elif suffix == ".txt":
            content = file_path.read_text(encoding="utf-8")
            return self.chunk_text(content, metadata)

        elif suffix == ".pdf":
            # Basic PDF text extraction (requires pypdf or similar)
            try:
                import pypdf
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    content = "\n\n".join(
                        page.extract_text() for page in reader.pages
                    )
                return self.chunk_text(content, metadata)
            except ImportError as e:
                logger.warning("pypdf not installed, cannot process PDF")
                raise ValueError("PDF support requires pypdf package") from e

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _split_by_headers(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Split markdown by headers.

        Args:
            markdown_text: Markdown text

        Returns:
            List of sections with header and content
        """
        sections = []
        current_header = ""
        current_content = []

        for line in markdown_text.split("\n"):
            # Check if line is a header (# Header or Header\n===)
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    sections.append({
                        "header": current_header,
                        "content": "\n".join(current_content)
                    })

                # Start new section
                current_header = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append({
                "header": current_header,
                "content": "\n".join(current_content)
            })

        return sections

    def _recursive_split(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """
        Recursively split text using separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        if not separators:
            # Base case: no more separators, split by character
            return self._split_by_length(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            # Rough token estimate (1 token ≈ 4 characters)
            estimated_tokens = len(split) // 4

            if estimated_tokens > self.chunk_size:
                # This split is too large, recurse with next separator
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                sub_chunks = self._recursive_split(split, remaining_separators)
                chunks.extend(sub_chunks)

            elif len(current_chunk) // 4 + estimated_tokens > self.chunk_size:
                # Adding this would exceed chunk size
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split

            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_by_length(self, text: str) -> List[str]:
        """
        Split text by length as a last resort.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunk_chars = self.chunk_size * 4  # Rough estimate
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap_chars

        return [chunk.strip() for chunk in chunks if chunk.strip()]
