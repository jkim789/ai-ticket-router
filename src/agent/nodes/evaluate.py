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
    logger.info("Evaluating KB confidence")

    if "agent_trace" not in state:
        state["agent_trace"] = []

    kb_results = state.get("kb_results", [])

    # If no results, route to human immediately
    if not kb_results:
        state["kb_confidence"] = 0.0
        state["action"] = "route_to_human"
        state["agent_trace"].append("evaluate_confidence: 0.00 (no KB results, routing to human)")
        logger.info("No KB results found, routing to human")
        return state

    # Calculate average similarity score
    avg_score = sum(r["score"] for r in kb_results) / len(kb_results)

    # If similarity is very low, route immediately
    if avg_score < 0.5:
        state["kb_confidence"] = avg_score
        state["action"] = "route_to_human"
        trace_msg = f"evaluate_confidence: {avg_score:.2f} (low similarity, routing to human)"
        state["agent_trace"].append(trace_msg)
        logger.info(f"Low similarity score {avg_score:.2f}, routing to human")
        return state

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
        state["kb_confidence"] = kb_confidence

        # Apply threshold
        if kb_confidence >= settings.CONFIDENCE_THRESHOLD:
            state["action"] = "auto_respond"
            trace_msg = f"evaluate_confidence: {kb_confidence:.2f} (above threshold, auto-responding)"
        else:
            state["action"] = "route_to_human"
            trace_msg = f"evaluate_confidence: {kb_confidence:.2f} (below threshold, routing to human)"

        state["agent_trace"].append(trace_msg)

        logger.info(
            f"Confidence evaluation: {kb_confidence:.2f}, action: {state['action']}"
        )

    except Exception as e:
        logger.error(f"Error in evaluate_confidence: {e}", exc_info=True)
        # Be conservative - route to human on error
        state["kb_confidence"] = 0.3
        state["action"] = "route_to_human"
        state["agent_trace"].append(f"evaluate_confidence: ERROR - {str(e)} (routing to human)")

    return state
