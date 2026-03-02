"""
Auto-response generation node.

Generates customer-facing responses using KB context and the LLM.
"""

import logging

from src.agent.state import TicketState
from src.config import get_llm_client, settings

logger = logging.getLogger(__name__)


async def generate_response(state: TicketState) -> TicketState:
    """
    Generate an automatic response using KB context.

    Creates a professional, helpful response based on the retrieved knowledge
    base articles. The response includes specific information from the KB and
    maintains a friendly, empathetic tone.

    Args:
        state: Current ticket state with raw_message and kb_results

    Returns:
        Updated state with auto_response and trace entry
    """
    logger.info("Generating auto-response")

    if "agent_trace" not in state:
        state["agent_trace"] = []

    kb_results = state.get("kb_results", [])

    # Format KB context
    kb_context = "\n\n".join([
        f"[Source {i+1}] {r['content']}"
        for i, r in enumerate(kb_results[:3])
    ])

    client = get_llm_client()

    prompt = f"""You are a helpful customer support agent for NovaTech Solutions.

Generate a response to this customer message using the provided knowledge base articles.

Customer message: "{state['raw_message']}"

Knowledge base context:
{kb_context}

Guidelines:
- Be friendly, professional, and empathetic
- Provide clear, actionable information
- Reference specific details from the KB articles
- Keep it concise but complete
- Use numbered lists for multi-step instructions
- End with an offer to help further if needed

Generate the response now:"""

    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )

        auto_response = response.choices[0].message.content.strip()
        state["auto_response"] = auto_response

        state["agent_trace"].append(
            f"generate_response: created {len(auto_response)} char response"
        )

        logger.info(f"Generated auto-response ({len(auto_response)} characters)")

    except Exception as e:
        logger.error(f"Error in generate_response: {e}", exc_info=True)
        state["auto_response"] = (
            "Thank you for contacting NovaTech Solutions. "
            "We're experiencing a temporary issue generating your response. "
            "A support team member will follow up with you shortly."
        )
        state["agent_trace"].append(f"generate_response: ERROR - {str(e)}")

    return state
