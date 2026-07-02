"""Deterministic intent classification.

Control flow (ask vs retrieve vs answer vs refuse) is kept rule-based and
independent of the LLM on purpose: the assignment explicitly warns that a
non-deterministic conversation shouldn't make the system fall apart, and an
LLM-classified control flow would make turn-to-turn behavior harder to
guarantee under the 8-turn budget. The LLM is only used downstream for
grounded text generation, never for deciding what kind of turn this is.
"""

import re
from typing import Literal

from app.services.state import ConversationState

QueryType = Literal["out_of_scope", "comparison", "refinement", "clarification", "recommendation"]

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above)",
    r"forget (all |the )?(your |previous )?instructions",
    r"you are now\b",
    r"pretend (to be|you are)",
    r"act as (a |an )?(?!recruiter|hiring manager|hr)",
    r"system prompt",
    r"reveal your (prompt|instructions|rules)",
    r"jailbreak",
    r"developer mode",
    r"do anything now",
    r"override your (rules|instructions)",
    r"new instructions?:",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bsalary\b", r"\bcompensation\b", r"\bpay\s+(range|scale|band)\b",
    r"salary negotiation", r"offer negotiation", r"\btax(es)?\b",
    r"visa sponsorship", r"work permit", r"termination policy", r"how to fire",
    r"labor law", r"labour law", r"legal advice", r"is it legal",
    r"employment contract", r"resignation letter", r"offer letter",
    r"interview questions? for", r"write a job (description|posting)",
    r"job posting", r"performance review template", r"how much should i pay",
]

COMPARISON_PATTERNS = [r"\bcompare\b", r"\bdifference between\b", r"\bvs\.?\b", r"\bversus\b"]

REFINEMENT_PATTERNS = [
    r"\badd\b", r"\bremove\b", r"\binclude\b", r"\bexclude\b", r"\binstead\b",
    r"\balso\b", r"\bfocus more on\b", r"\bless technical\b", r"\bmore technical\b",
    r"\bshorter\b", r"\blonger\b", r"\bnarrow (it )?down\b", r"\bonly (the )?top\b",
    r"\bkeep only\b", r"\breplace\b", r"\bswap\b", r"\bchange\b", r"\bprioriti[sz]e\b",
    r"\bremote\b", r"\badaptive\b", r"\bwithout\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_out_of_scope(text: str) -> bool:
    return _matches_any(text, INJECTION_PATTERNS) or _matches_any(text, OUT_OF_SCOPE_PATTERNS)


def classify(state: ConversationState) -> QueryType:
    text = state.latest_user_message

    if is_out_of_scope(text):
        return "out_of_scope"

    if _matches_any(text, COMPARISON_PATTERNS):
        return "comparison"

    if state.has_prior_recommendation and _matches_any(text, REFINEMENT_PATTERNS):
        return "refinement"

    if state.recommendation_ready:
        return "recommendation"

    return "clarification"
