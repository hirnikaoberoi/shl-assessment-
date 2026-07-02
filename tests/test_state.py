import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.state import build_state


def _msgs(turns):
    return [{"role": r, "content": c} for r, c in turns]


def test_extracts_prior_recommendations_from_marker():
    turns = [
        ("user", "Hiring Java developers for a coding assessment for selection."),
        ("assistant", "Here you go.\n\nShortlisted assessments: Java 8 (New) | Verify G+"),
        ("user", "Add a personality test too"),
    ]
    state = build_state(_msgs(turns))
    assert state.has_prior_recommendation is True
    assert state.prior_recommended_names == ["Java 8 (New)", "Verify G+"]


def test_no_marker_means_no_prior_recommendation():
    turns = [
        ("user", "I need an assessment"),
        ("assistant", "Sure -- what role is this for?"),
    ]
    state = build_state(_msgs(turns))
    assert state.has_prior_recommendation is False
    assert state.prior_recommended_names == []


def test_uses_most_recent_marker_when_multiple_exist():
    turns = [
        ("user", "Hiring Java developers for a coding assessment for selection."),
        ("assistant", "Here you go.\n\nShortlisted assessments: Java 8 (New)"),
        ("user", "Also add personality"),
        ("assistant", "Updated.\n\nShortlisted assessments: Java 8 (New) | OPQ32r"),
    ]
    state = build_state(_msgs(turns))
    assert state.prior_recommended_names == ["Java 8 (New)", "OPQ32r"]


def test_empty_history_is_safe():
    state = build_state([])
    assert state.latest_user_message == ""
    assert state.recommendation_ready is False
    assert state.turn_count == 0
