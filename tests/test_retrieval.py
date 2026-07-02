import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.retrieval import BM25Index, tokenize

FAKE_DOCS = [
    {
        "name": "Java 8 (New)",
        "search_text": "java 8 knowledge skills programming coding developer",
    },
    {
        "name": "OPQ32r",
        "search_text": "opq32r personality behaviour occupational questionnaire leadership manager",
    },
    {
        "name": "Verify Interactive G+",
        "search_text": "verify interactive g+ cognitive reasoning ability aptitude numerical",
    },
    {
        "name": "Customer Service Simulation",
        "search_text": "customer service simulation call center support communication",
    },
]


def test_tokenize_strips_stopwords():
    tokens = tokenize("We need an assessment for a Java developer")
    assert "java" in tokens
    assert "developer" in tokens
    assert "we" not in tokens
    assert "need" not in tokens


def test_relevant_document_ranks_first():
    index = BM25Index(FAKE_DOCS)
    results = index.search("Java developer coding test", top_k=4)
    assert results[0]["name"] == "Java 8 (New)"


def test_synonym_expansion_improves_recall():
    index = BM25Index(FAKE_DOCS)
    # 'programmer' is not literally in any search_text except via synonym
    # expansion mapping it back to 'developer'/'coding'/'engineer'.
    results = index.search("hiring a programmer", top_k=4)
    names = [r["name"] for r in results]
    assert "Java 8 (New)" in names


def test_no_matching_terms_returns_empty():
    index = BM25Index(FAKE_DOCS)
    results = index.search("zzzznonexistenttermxyz", top_k=4)
    assert results == []


def test_leadership_query_ranks_personality_assessment():
    index = BM25Index(FAKE_DOCS)
    results = index.search("executive leadership assessment for senior managers", top_k=4)
    assert results[0]["name"] == "OPQ32r"


def test_top_k_is_respected():
    index = BM25Index(FAKE_DOCS)
    results = index.search("assessment test", top_k=2)
    assert len(results) <= 2
