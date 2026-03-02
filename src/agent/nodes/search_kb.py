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
    logger.info(
        "search_kb_start",
        extra={"request_id": state.get("request_id")},
    )

    try:
        # Search with metadata filtering by intent category
        results = await vectorstore.search(
            query=state["raw_message"],
            category=state.get("intent"),
            limit=5
        )

        top_score = results[0]["score"] if results else 0.0
        trace_msg = f"search_kb: found {len(results)} results, top score {top_score:.3f}"
        agent_trace = state.get("agent_trace", [])
        agent_trace = [*agent_trace, trace_msg]

        logger.info(
            "search_kb_complete",
            extra={
                "request_id": state.get("request_id"),
            },
        )

        # Format results
        kb_results = [
            {
                "content": doc["content"],
                "score": doc["score"],
                "metadata": doc.get("metadata", {}),
            }
            for doc in results
        ]
        return {
            "kb_results": kb_results,
            "agent_trace": agent_trace,
        }

    except Exception as e:
        logger.error("Error in search_knowledge_base", exc_info=True)
        agent_trace = state.get("agent_trace", [])
        agent_trace = [*agent_trace, f"search_kb: ERROR - {str(e)}"]
        return {
            "kb_results": [],
            "agent_trace": agent_trace,
        }
