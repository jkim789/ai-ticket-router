"""Unit tests for routing logic."""

from src.agent.nodes.route import INTENT_TO_TEAM, calculate_priority


def test_calculate_priority_critical_urgency():
    """Test that critical urgency always results in critical priority."""
    assert calculate_priority("neutral", "critical") == "critical"
    assert calculate_priority("positive", "critical") == "critical"


def test_calculate_priority_angry_sentiment():
    """Test that angry sentiment elevates priority."""
    assert calculate_priority("angry", "medium") == "critical"
    assert calculate_priority("angry", "high") == "critical"
    assert calculate_priority("angry", "low") == "high"


def test_calculate_priority_high_urgency_negative():
    """Test high urgency with negative sentiment."""
    assert calculate_priority("negative", "high") == "high"


def test_calculate_priority_standard():
    """Test standard priority mapping."""
    assert calculate_priority("neutral", "low") == "low"
    assert calculate_priority("neutral", "medium") == "medium"
    assert calculate_priority("positive", "medium") == "medium"


def test_intent_to_team_mapping():
    """Test intent to team mapping."""
    assert INTENT_TO_TEAM["billing"] == "billing_team"
    assert INTENT_TO_TEAM["technical"] == "engineering"
    assert INTENT_TO_TEAM["shipping"] == "logistics"
    assert INTENT_TO_TEAM["complaint"] == "escalation"
    assert INTENT_TO_TEAM["refund"] == "billing_team"
    assert INTENT_TO_TEAM["general"] == "general_support"
