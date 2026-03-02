"""
LangGraph state machine assembly.

This module compiles all agent nodes into a complete workflow graph.
"""

import logging
from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes.classify import classify_intent
from src.agent.nodes.evaluate import evaluate_confidence
from src.agent.nodes.respond import generate_response
from src.agent.nodes.route import route_to_human
from src.agent.nodes.search_kb import search_knowledge_base
from src.agent.state import TicketState

logger = logging.getLogger(__name__)


def should_auto_respond(state: TicketState) -> str:
    """
    Routing function for conditional edge after evaluate_confidence.

    Args:
        state: Current ticket state with action field

    Returns:
        Next node name: "generate_response" or "route_to_human"
    """
    action = state.get("action", "route_to_human")
    if action == "auto_respond":
        return "generate_response"
    return "route_to_human"


def build_graph(vectorstore: Any) -> StateGraph:
    """
    Build and compile the complete LangGraph workflow.

    The graph flow:
    1. classify_intent - Classify intent, sentiment, urgency
    2. search_kb - Search knowledge base with intent filtering
    3. evaluate_confidence - Assess KB result quality
    4. [Conditional] If confident -> generate_response, else -> route_to_human

    Args:
        vectorstore: ChromaDB vectorstore instance to inject into nodes

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Building LangGraph workflow")

    # Create the graph
    workflow = StateGraph(TicketState)

    # Add nodes (search_kb needs vectorstore injected)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("search_kb", partial(search_knowledge_base, vectorstore=vectorstore))
    workflow.add_node("evaluate_confidence", evaluate_confidence)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("route_to_human", route_to_human)

    # Define edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "search_kb")
    workflow.add_edge("search_kb", "evaluate_confidence")

    # Conditional edge based on confidence evaluation
    workflow.add_conditional_edges(
        "evaluate_confidence",
        should_auto_respond,
        {
            "generate_response": "generate_response",
            "route_to_human": "route_to_human"
        }
    )

    # Both paths end after their respective nodes
    workflow.add_edge("generate_response", END)
    workflow.add_edge("route_to_human", END)

    # Compile
    graph = workflow.compile()

    logger.info("LangGraph workflow compiled successfully")

    return graph
