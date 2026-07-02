"""Turns a message history into the final ChatResponse.

The critical rule enforced here (missing in earlier iterations of this
project): `recommendations` is only ever populated on "recommendation" and
"refinement" turns. It is structurally impossible for a clarification,
comparison, or out-of-scope turn to end up with a non-empty recommendations
array -- that gate is applied here in code, not left to prompt instructions
the model could ignore.
"""

import difflib

from app.config import MAX_RECOMMENDATIONS, RETRIEVAL_TOP_K
from app.services import catalog, llm
from app.services.classifier import QueryType, classify
from app.services.retrieval import semantic_search
from app.services.state import RECOMMENDATION_MARKER, ConversationState, build_state

OUT_OF_SCOPE_REPLY = (
    "I can only help with finding and comparing SHL assessments. I'm not able to help with "
    "that -- if you'd like, tell me about the role or skills you're hiring for and I can "
    "suggest relevant SHL assessments."
)

LLM_ERROR_REPLY = (
    "I'm having trouble reaching the recommendation engine right now. Could you try again, "
    "or rephrase what you're looking for?"
)

RETRIEVAL_QUERY_TYPES: set[QueryType] = {"recommendation", "refinement", "clarification"}


def _retrieval_query(state: ConversationState) -> str:
    return f"{state.all_user_text} {state.latest_user_message}".strip()


def _resolve_names_to_catalog(names: list[str], candidates: list[dict]) -> list[dict]:
    """Exact match first; if the model paraphrased slightly, fall back to a
    close-match search restricted to the retrieved candidate pool only (never
    the full catalog) so a near-miss can't resolve to an unrelated item."""
    resolved = []
    seen_names = set()
    candidate_by_name = {c["name"].strip().lower(): c for c in candidates}

    for name in names:
        if len(resolved) >= MAX_RECOMMENDATIONS:
            break
        key = name.strip().lower()

        item = candidate_by_name.get(key) or catalog.find_by_exact_name(name)

        if item is None and candidates:
            close = difflib.get_close_matches(
                key, list(candidate_by_name.keys()), n=1, cutoff=0.85
            )
            if close:
                item = candidate_by_name[close[0]]

        if item is None:
            continue
        if item["name"] in seen_names:
            continue
        seen_names.add(item["name"])
        resolved.append(item)

    return resolved


def _to_recommendation_dicts(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        out.append(
            {
                "name": item["name"],
                "url": item["url"],
                "test_type": catalog.primary_test_type_code(item),
            }
        )
    return out


def _append_marker(reply: str, names: list[str]) -> str:
    if not names:
        return reply
    return f"{reply}\n\n{RECOMMENDATION_MARKER} {' | '.join(names)}"


def handle_chat(messages: list[dict]) -> dict:
    state = build_state(messages)
    query_type = classify(state)

    if query_type == "out_of_scope":
        return {"reply": OUT_OF_SCOPE_REPLY, "recommendations": [], "end_of_conversation": False}

    candidates: list[dict] = []
    if query_type == "comparison":
        # Comparisons are about specific named assessments, not a broad
        # relevance search -- pull directly by name instead of BM25.
        candidates = _lookup_named_assessments(state.latest_user_message)
    elif query_type in RETRIEVAL_QUERY_TYPES:
        candidates = semantic_search(_retrieval_query(state), top_k=RETRIEVAL_TOP_K)

    prompt_builders = {
        "clarification": llm.build_clarification_prompt,
        "recommendation": llm.build_recommendation_prompt,
        "refinement": llm.build_refinement_prompt,
        "comparison": llm.build_comparison_prompt,
    }
    build_prompt = prompt_builders[query_type]
    user_prompt = build_prompt(state, candidates)

    try:
        result = llm.call_llm(user_prompt)
    except Exception:
        return {"reply": LLM_ERROR_REPLY, "recommendations": [], "end_of_conversation": False}

    reply = result["reply"]

    # Structural gate: only these two turn types are allowed to carry a
    # non-empty recommendations array, regardless of what the model returned.
    if query_type not in ("recommendation", "refinement"):
        return {"reply": reply, "recommendations": [], "end_of_conversation": False}

    resolved_items = _resolve_names_to_catalog(result["recommended_assessment_names"], candidates)

    if not resolved_items and candidates:
        # The model committed to recommending but didn't name anything we
        # could verify against the catalog -- fall back to the top retrieved
        # candidates so the user still gets a usable shortlist this turn
        # instead of an empty one.
        resolved_items = candidates[: min(5, MAX_RECOMMENDATIONS)]

    recommendations = _to_recommendation_dicts(resolved_items)
    reply = _append_marker(reply, [item["name"] for item in resolved_items])

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": len(recommendations) > 0,
    }


def _lookup_named_assessments(text: str) -> list[dict]:
    all_items = catalog.load_catalog()
    text_lower = text.lower()
    matches = [item for item in all_items if item["name"].lower() in text_lower]
    if matches:
        return matches[:RETRIEVAL_TOP_K]
    # No exact name match (e.g. user used an acronym or partial name) --
    # fall back to lexical retrieval so the model still has *something*
    # grounded to compare, and will say so if it can't confidently match.
    return semantic_search(text, top_k=RETRIEVAL_TOP_K)
