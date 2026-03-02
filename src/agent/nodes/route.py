"""
Human routing node.

Creates structured handoff packets for human agents.
"""

import logging

from src.agent.state import TicketState
from src.config import get_llm_client, settings

logger = logging.getLogger(__name__)


# Intent to team mapping
INTENT_TO_TEAM = {
    "billing": "billing_team",
    "technical": "engineering",
    "shipping": "logistics",
    "complaint": "escalation",
    "refund": "billing_team",
    "general": "general_support"
}


def calculate_priority(sentiment: str, urgency: str) -> str:
    """
    Calculate routing priority based on sentiment and urgency.

    Args:
        sentiment: Customer sentiment (positive, neutral, negative, angry)
        urgency: Issue urgency (low, medium, high, critical)

    Returns:
        Priority level (low, medium, high, critical)
    """
    # Critical urgency always maps to critical priority
    if urgency == "critical":
        return "critical"

    # Angry sentiment elevates priority
    if sentiment == "angry":
        if urgency in ("high", "medium"):
            return "critical"
        return "high"

    # High urgency with negative sentiment
    if urgency == "high" and sentiment == "negative":
        return "high"

    # Direct mapping for other cases
    return urgency


async def route_to_human(state: TicketState) -> TicketState:
    """
    Create a structured handoff packet for human agents.

    This node prepares all the context a human agent needs to efficiently
    handle the ticket, including:
    - Team assignment based on intent
    - Priority based on sentiment and urgency
    - Summary of the issue
    - Relevant KB articles for agent reference

    Args:
        state: Current ticket state with all classification and KB results

    Returns:
        Updated state with routing information and trace entry
    """
    logger.info(
        "route_to_human_start",
        extra={"request_id": state.get("request_id")},
    )

    intent = state.get("intent", "general")
    sentiment = state.get("sentiment", "neutral")
    urgency = state.get("urgency", "medium")

    # Determine team
    team = INTENT_TO_TEAM.get(intent, "general_support")

    # Calculate priority
    priority = calculate_priority(sentiment, urgency)

    # Generate context summary using LLM
    client = get_llm_client()

    kb_results = state.get("kb_results", [])
    kb_context = "\n".join([
        f"- {r['content'][:200]}... (relevance: {r['score']:.2f})"
        for r in kb_results[:3]
    ]) if kb_results else "No relevant KB articles found."

    prompt = f"""Create a brief summary for a human support agent taking over this ticket.

Customer message: "{state['raw_message']}"
Intent: {intent}
Sentiment: {sentiment}
Urgency: {urgency}

Relevant KB articles:
{kb_context}

Write a 2-3 sentence summary that gives the agent context about:
1. What the customer needs
2. Why it requires human attention
3. Any relevant KB info they should reference

Keep it concise and actionable."""

    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        summary = response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        summary = f"Customer inquiry regarding {intent}. Requires human attention due to complexity or low KB confidence."

    # Format relevant KB articles for agent reference
    relevant_articles = [
        {
            "title": r.get("metadata", {}).get("title", "KB Article"),
            "relevance_score": r["score"],
            "content_preview": r["content"][:200]
        }
        for r in kb_results[:3]
    ] if kb_results else []

    trace_msg = f"route_to_human: {team}, priority {priority}"
    agent_trace = state.get("agent_trace", [])
    agent_trace = [*agent_trace, trace_msg]

    logger.info(
        "route_to_human_complete",
        extra={
            "request_id": state.get("request_id"),
            "action": "route_to_human",
        },
    )

    # Create routing packet
    return {
        "routing": {
            "team": team,
            "priority": priority,
            "summary": summary,
            "context": relevant_articles,
        },
        "agent_trace": agent_trace,
    }
