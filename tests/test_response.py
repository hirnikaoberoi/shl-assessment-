import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import response

FAKE_CATALOG = [
    {
        "entity_id": "1",
        "name": "Java 8 (New)",
        "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
        "description": "Measures knowledge of core Java 8 features.",
        "job_levels": ["Mid-Professional"],
        "test_types": [{"code": "K", "name": "Knowledge & Skills"}],
        "duration_minutes": 30,
        "remote": True,
        "adaptive": False,
        "languages": ["English (USA)"],
        "search_text": "java 8 knowledge skills",
    },
    {
        "entity_id": "2",
        "name": "OPQ32r",
        "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
        "description": "Occupational personality questionnaire.",
        "job_levels": ["Manager", "Executive"],
        "test_types": [{"code": "P", "name": "Personality & Behaviour"}],
        "duration_minutes": 25,
        "remote": True,
        "adaptive": False,
        "languages": ["English (USA)"],
        "search_text": "opq32r personality behaviour",
    },
    {
        "entity_id": "3",
        "name": "Global Skills Assessment (GSA)",
        "url": "https://www.shl.com/products/product-catalog/view/gsa/",
        "description": "Broad measure of workplace skills.",
        "job_levels": ["General Population"],
        "test_types": [{"code": "A", "name": "Ability & Aptitude"}],
        "duration_minutes": 36,
        "remote": True,
        "adaptive": False,
        "languages": ["English (USA)"],
        "search_text": "global skills assessment gsa ability aptitude",
    },
]


def _msgs(turns):
    return [{"role": r, "content": c} for r, c in turns]


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_clarification_never_has_recommendations(mock_llm, _search, _load):
    mock_llm.return_value = {
        "reply": "What role and what should the assessment focus on?",
        "recommended_assessment_names": ["Java 8 (New)"],  # model misbehaving on purpose
    }
    result = response.handle_chat(_msgs([("user", "I need an assessment")]))
    assert result["recommendations"] == []
    assert result["end_of_conversation"] is False


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_comparison_never_has_recommendations(mock_llm, _search, _load):
    mock_llm.return_value = {
        "reply": "OPQ32r measures personality; GSA measures broad workplace skills.",
        "recommended_assessment_names": ["OPQ32r", "Global Skills Assessment (GSA)"],
    }
    result = response.handle_chat(_msgs([("user", "Compare OPQ32r and GSA")]))
    assert result["recommendations"] == []
    assert result["end_of_conversation"] is False


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_out_of_scope_never_calls_llm(mock_llm, _search, _load):
    result = response.handle_chat(_msgs([("user", "What salary should I offer a Java developer?")]))
    assert result["recommendations"] == []
    assert result["end_of_conversation"] is False
    mock_llm.assert_not_called()


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_recommendation_resolves_exact_catalog_names(mock_llm, _search, _load):
    mock_llm.return_value = {
        "reply": "Here are some options.",
        "recommended_assessment_names": ["Java 8 (New)", "Made Up Assessment That Does Not Exist"],
    }
    result = response.handle_chat(_msgs([
        ("user", "Hiring mid-level backend Java engineers for a coding assessment for selection."),
    ]))
    names = [r["name"] for r in result["recommendations"]]
    assert names == ["Java 8 (New)"]
    assert result["end_of_conversation"] is True
    assert all(r["url"].startswith("https://www.shl.com/") for r in result["recommendations"])


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_recommendation_falls_back_to_candidates_if_llm_names_nothing_real(mock_llm, _search, _load):
    mock_llm.return_value = {
        "reply": "Here are some options.",
        "recommended_assessment_names": ["Totally Invented Name"],
    }
    result = response.handle_chat(_msgs([
        ("user", "Hiring mid-level backend Java engineers for a coding assessment for selection."),
    ]))
    assert len(result["recommendations"]) > 0
    assert result["end_of_conversation"] is True


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_llm_failure_returns_valid_schema(mock_llm, _search, _load):
    mock_llm.side_effect = RuntimeError("groq unreachable")
    result = response.handle_chat(_msgs([
        ("user", "Hiring mid-level backend Java engineers for a coding assessment for selection."),
    ]))
    assert result["recommendations"] == []
    assert result["end_of_conversation"] is False
    assert isinstance(result["reply"], str) and result["reply"]


@patch("app.services.response.catalog.load_catalog", return_value=FAKE_CATALOG)
@patch("app.services.response.semantic_search", return_value=FAKE_CATALOG)
@patch("app.services.response.llm.call_llm")
def test_refinement_carries_forward_after_marker(mock_llm, _search, _load):
    mock_llm.return_value = {
        "reply": "Updated shortlist with a personality assessment added.",
        "recommended_assessment_names": ["Java 8 (New)", "OPQ32r"],
    }
    turns = [
        ("user", "Hiring mid-level backend Java engineers for a coding assessment for selection."),
        ("assistant", "Here you go.\n\nShortlisted assessments: Java 8 (New)"),
        ("user", "Actually also add a personality assessment"),
    ]
    result = response.handle_chat(_msgs(turns))
    names = [r["name"] for r in result["recommendations"]]
    assert "OPQ32r" in names
    assert result["end_of_conversation"] is True
