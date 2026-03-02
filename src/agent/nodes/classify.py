"""
Intent classification node using Groq API.

Classifies customer messages into intent, sentiment, and urgency categories.
"""

import json
import logging

from src.agent.state import TicketState
from src.config import get_llm_client, settings

logger = logging.getLogger(__name__)


async def classify_intent(state: TicketState) -> TicketState:
    """
    Classify the intent, sentiment, and urgency of the customer message.

    This node uses the LLM to analyze the customer's message and extract:
    - Intent: The primary category of the support request
    - Sentiment: The emotional tone of the message
    - Urgency: How time-sensitive the issue is

    Args:
        state: Current ticket state containing raw_message

    Returns:
        Updated state with intent, sentiment, urgency, and trace entry
    """
    logger.info(
        "classify_intent_start",
        extra={"request_id": state.get("request_id")},
    )

    client = get_llm_client()

    prompt = f"""Analyze the following customer support message and classify it.

Customer message: "{state['raw_message']}"

Provide your analysis in JSON format with these exact fields:
{{
  "intent": "<billing|technical|shipping|general|complaint|refund>",
  "sentiment": "<positive|neutral|negative|angry>",
  "urgency": "<low|medium|high|critical>",
  "reasoning": "<brief explanation of your classification>"
}}

Intent definitions:
- billing: Payment issues, invoices, subscription questions
- technical: Software bugs, feature issues, account problems
- shipping: Delivery questions, tracking, lost packages
- general: General inquiries, business info, how-to questions
- complaint: Customer complaints, dissatisfaction, service issues
- refund: Refund requests, money-back requests

Sentiment definitions:
- positive: Happy, satisfied, thankful tone
- neutral: Matter-of-fact, no strong emotion
- negative: Frustrated, disappointed, dissatisfied
- angry: Very upset, demanding, using strong language

Urgency definitions:
- low: No time pressure, general question
- medium: Wants resolution soon, but not blocking
- high: Needs quick resolution, affecting their business/use
- critical: Urgent, losing money, severe impact, angry customer

Respond ONLY with the JSON object, no other text."""

    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

        # Parse the JSON response
        content = response.choices[0].message.content
        result = json.loads(content)

        trace_msg = (
            f"classify_intent: {result['intent']} "
            f"(sentiment: {result['sentiment']}, urgency: {result['urgency']}) - "
            f"{result['reasoning']}"
        )
        agent_trace = state.get("agent_trace", [])
        agent_trace = [*agent_trace, trace_msg]

        logger.info(
            "classify_intent_complete",
            extra={
                "request_id": state.get("request_id"),
                "action": "classify",
            },
        )

        return {
            "intent": result["intent"],
            "sentiment": result["sentiment"],
            "urgency": result["urgency"],
            "model_used": settings.LLM_MODEL,
            "agent_trace": agent_trace,
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse classification JSON", exc_info=True)
        agent_trace = state.get("agent_trace", [])
        agent_trace = [
            *agent_trace,
            f"classify_intent: ERROR - {str(e)} (fallback to general)",
        ]
        return {
            "intent": "general",
            "sentiment": "neutral",
            "urgency": "medium",
            "agent_trace": agent_trace,
        }

    except Exception as e:
        logger.error("Error in classify_intent", exc_info=True)
        agent_trace = state.get("agent_trace", [])
        agent_trace = [
            *agent_trace,
            f"classify_intent: ERROR - {str(e)}",
        ]
        return {
            "intent": "general",
            "sentiment": "neutral",
            "urgency": "medium",
            "agent_trace": agent_trace,
        }
