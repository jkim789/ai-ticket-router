"""
Knowledge base search node.

Performs semantic search against ChromaDB to find relevant support articles.
"""

import logging
from typing import Any

from src.agent.state import TicketState

logger = logging.getLogger(__name__)


async def search_knowledge_base(
    state: TicketState,
    vectorstore: Any  # Will be injected via partial
) -> TicketState:
    """
    Search the knowledge base for relevant documents.

    Performs semantic search using the customer's message and filters by
    the classified intent category for better precision.

    Args:
        state: Current ticket state with raw_message and intent
        vectorstore: ChromaDB vectorstore instance

    Returns:
        Updated state with kb_results and trace entry
    """
    logger.info(f"Searching knowledge base for intent: {state.get('intent')}")

    if "agent_trace" not in state:
        state["agent_trace"] = []

    try:
        # Search with metadata filtering by intent category
        results = await vectorstore.search(
            query=state["raw_message"],
            category=state.get("intent"),
            limit=5
        )

        # Format results
        state["kb_results"] = [
            {
                "content": doc["content"],
                "score": doc["score"],
                "metadata": doc.get("metadata", {})
            }
            for doc in results
        ]

        top_score = results[0]["score"] if results else 0.0
        trace_msg = f"search_kb: found {len(results)} results, top score {top_score:.3f}"
        state["agent_trace"].append(trace_msg)

        logger.info(f"Found {len(results)} KB results")

    except Exception as e:
        logger.error(f"Error in search_knowledge_base: {e}", exc_info=True)
        state["kb_results"] = []
        state["agent_trace"].append(f"search_kb: ERROR - {str(e)}")

    return state
