"""
Seed the knowledge base with documents from data/seed/knowledge_base/.

This script loads markdown documents, chunks them, and ingests them into ChromaDB.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker import DocumentChunker
from src.rag.vectorstore import VectorStore
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Category mapping from filename to intent category
CATEGORY_MAPPING = {
    "billing.md": "billing",
    "technical.md": "technical",
    "shipping.md": "shipping",
    "returns.md": "refund",  # Returns maps to refund intent
    "general.md": "general"
}


async def seed_knowledge_base():
    """
    Load seed documents into ChromaDB.

    Reads all markdown files from data/seed/knowledge_base/, chunks them,
    and loads them into the vector store with appropriate metadata.
    """
    logger.info("Starting knowledge base seeding...")

    # Initialize vectorstore
    vectorstore = VectorStore()
    await vectorstore.initialize()

    # Initialize chunker
    chunker = DocumentChunker(
        chunk_size=512,
        chunk_overlap=50
    )

    # Find all seed documents
    seed_dir = Path(__file__).parent.parent / "data" / "seed" / "knowledge_base"

    if not seed_dir.exists():
        logger.error(f"Seed directory not found: {seed_dir}")
        return

    markdown_files = list(seed_dir.glob("*.md"))

    if not markdown_files:
        logger.error(f"No markdown files found in {seed_dir}")
        return

    logger.info(f"Found {len(markdown_files)} seed documents")

    # Process each file
    all_chunks = []
    total_chunks = 0

    for md_file in markdown_files:
        logger.info(f"Processing {md_file.name}...")

        # Determine category
        category = CATEGORY_MAPPING.get(md_file.name, "general")

        # Chunk the file
        chunks = chunker.chunk_file(md_file, category=category)

        logger.info(f"  Created {len(chunks)} chunks from {md_file.name}")
        all_chunks.extend(chunks)
        total_chunks += len(chunks)

    logger.info(f"Total chunks created: {total_chunks}")

    # Prepare data for ChromaDB
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    ids = [f"seed_{i}" for i in range(len(all_chunks))]

    # Add to vectorstore (ChromaDB handles embedding automatically)
    logger.info("Adding documents to ChromaDB...")

    try:
        await vectorstore.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info("✓ Knowledge base seeding complete!")

        # Print stats
        stats = await vectorstore.get_stats()
        logger.info(f"Total documents in KB: {stats['total_documents']}")

        # Count by category
        category_counts = {}
        for metadata in metadatas:
            cat = metadata.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        logger.info("Documents by category:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"  {category}: {count}")

    except Exception as e:
        logger.error(f"Error seeding knowledge base: {e}", exc_info=True)
        raise

    finally:
        await vectorstore.close()


async def test_search():
    """
    Test the knowledge base with sample queries.

    This function is optional and demonstrates that seeding worked.
    """
    logger.info("\n" + "="*60)
    logger.info("Testing knowledge base search...")
    logger.info("="*60)

    vectorstore = VectorStore()
    await vectorstore.initialize()

    test_queries = [
        ("How do I reset my password?", "technical"),
        ("What is your refund policy?", "billing"),
        ("Where is my package?", "shipping"),
        ("Can I return an item?", "refund"),
        ("What are your business hours?", "general"),
    ]

    for query, expected_category in test_queries:
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Expected category: {expected_category}")

        results = await vectorstore.search(query=query, limit=3)

        if results:
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score = result["score"]
                category = result["metadata"].get("category", "unknown")
                content_preview = result["content"][:100] + "..."

                logger.info(f"  {i}. Score: {score:.3f}, Category: {category}")
                logger.info(f"     {content_preview}")
        else:
            logger.warning("  No results found!")

    await vectorstore.close()
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the NovaTech knowledge base")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries after seeding"
    )
    args = parser.parse_args()

    # Run seeding
    asyncio.run(seed_knowledge_base())

    # Optionally run tests
    if args.test:
        asyncio.run(test_search())

    logger.info("\n✅ All done!")
