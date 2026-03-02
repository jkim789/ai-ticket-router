"""
Confidence evaluation node.

Evaluates the quality of KB search results to decide if auto-response is appropriate.
"""

import json
import logging

from src.agent.state import TicketState
from src.config import get_llm_client, settings

logger = logging.getLogger(__name__)


async def evaluate_confidence(state: TicketState) -> TicketState:
    """
    Evaluate if KB results are sufficient for auto-response.

    This node analyzes the quality and relevance of retrieved KB articles
    to determine if we can confidently auto-respond or should route to a human.

    The evaluation considers:
    - Similarity scores of retrieved documents
    - Whether the documents actually answer the question
    - Completeness of the information

    Args:
        state: Current ticket state with kb_results

    Returns:
        Updated state with kb_confidence, action, and trace entry
    """
    logger.info(
        "evaluate_confidence_start",
        extra={"request_id": state.get("request_id")},
    )

    kb_results = state.get("kb_results", [])

    # If no results, route to human immediately
    if not kb_results:
        logger.info(
            "evaluate_confidence_no_results",
            extra={"request_id": state.get("request_id")},
        )
        agent_trace = state.get("agent_trace", [])
        agent_trace = [
            *agent_trace,
            "evaluate_confidence: 0.00 (no KB results, routing to human)",
        ]
        return {
            "kb_confidence": 0.0,
            "action": "route_to_human",
            "agent_trace": agent_trace,
        }

    # Calculate average similarity score
    avg_score = sum(r["score"] for r in kb_results) / len(kb_results)

    # If similarity is very low, route immediately
    if avg_score < 0.5:
        trace_msg = f"evaluate_confidence: {avg_score:.2f} (low similarity, routing to human)"
        agent_trace = state.get("agent_trace", [])
        agent_trace = [*agent_trace, trace_msg]
        logger.info(
            "evaluate_confidence_low_similarity",
            extra={"request_id": state.get("request_id")},
        )
        return {
            "kb_confidence": avg_score,
            "action": "route_to_human",
            "agent_trace": agent_trace,
        }

    # Use LLM to evaluate quality
    client = get_llm_client()

    # Format KB results for the prompt
    kb_context = "\n\n".join([
        f"Article {i+1} (relevance: {r['score']:.2f}):\n{r['content']}"
        for i, r in enumerate(kb_results[:3])
    ])

    prompt = f"""Evaluate if the knowledge base articles can answer the customer's question.

Customer question: "{state['raw_message']}"

Retrieved KB articles:
{kb_context}

Analyze whether these articles provide enough information to generate a complete, accurate response.

Consider:
- Do the articles directly address the customer's question?
- Is the information complete and actionable?
- Are there any gaps or ambiguities?

Respond in JSON format:
{{
  "can_answer": <true|false>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}

Respond ONLY with the JSON object."""

    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        kb_confidence = result["confidence"]

        # Apply threshold
        if kb_confidence >= settings.CONFIDENCE_THRESHOLD:
            action = "auto_respond"
            trace_msg = f"evaluate_confidence: {kb_confidence:.2f} (above threshold, auto-responding)"
        else:
            action = "route_to_human"
            trace_msg = f"evaluate_confidence: {kb_confidence:.2f} (below threshold, routing to human)"

        agent_trace = state.get("agent_trace", [])
        agent_trace = [*agent_trace, trace_msg]

        logger.info(
            "evaluate_confidence_complete",
            extra={
                "request_id": state.get("request_id"),
                "action": action,
            },
        )

        return {
            "kb_confidence": kb_confidence,
            "action": action,
            "agent_trace": agent_trace,
        }

    except Exception as e:
        logger.error("Error in evaluate_confidence", exc_info=True)
        agent_trace = state.get("agent_trace", [])
        agent_trace = [
            *agent_trace,
            f"evaluate_confidence: ERROR - {str(e)} (routing to human)",
        ]
        # Be conservative - route to human on error
        return {
            "kb_confidence": 0.3,
            "action": "route_to_human",
            "agent_trace": agent_trace,
        }
