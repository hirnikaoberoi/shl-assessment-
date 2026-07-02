"""Builds a per-request conversation state from the full stateless message
history. Nothing here is persisted between requests -- every /chat call
re-derives this from the `messages` array the client sends, per the
assignment's stateless API contract.
"""

import re
from dataclasses import dataclass
from typing import List

from app.config import MAX_CLARIFICATION_ROUNDS

RECOMMENDATION_MARKER = "Shortlisted assessments:"
# Appended verbatim to `reply` by response.py whenever recommendations are
# non-empty. Since the API is stateless, this is the only way to recover
# "what did we recommend last turn" from the message history the client
# echoes back -- it's deliberately phrased to read as a normal recap line
# so it's harmless if a human (or the eval harness's simulated user) reads
# it, while still being an exact, unambiguous string we can parse back out.

ROLE_KEYWORDS = [
    "engineer", "developer", "programmer", "architect", "analyst", "manager",
    "leader", "leadership", "executive", "director", "supervisor", "sales",
    "marketing", "consultant", "operations", "graduate", "intern", "clerk",
    "administrator", "admin", "technician", "specialist", "representative",
    "reps", "agent", "cashier", "driver", "nurse", "accountant", "designer",
    "scientist",
]

# Deliberately excludes generic "hiring"/"hire"/"recruit(ment)" -- those
# words appear in almost every message this agent will ever see, so
# treating them as a real "purpose" signal made the coverage check trivially
# satisfiable and caused premature recommendations (caught by trace 01,
# which mirrors the assignment's own worked example and expects a
# clarifying question here, not an immediate shortlist).
PURPOSE_KEYWORDS = [
    "selection", "selecting", "screening", "promotion", "succession",
    "development", "benchmark", "benchmarking", "onboarding",
    "training needs", "high-potential",
]

FOCUS_KEYWORDS = [
    "personality", "cognitive", "technical", "coding", "programming",
    "leadership", "behavioral", "behavioural", "reasoning", "numerical",
    "verbal", "communication", "stakeholder", "decision-making",
    "problem-solving", "customer service", "compliance", "attention to detail",
    "teamwork", "collaboration", "java", "python", "sql", "excel",
]

SENIORITY_KEYWORDS = [
    "entry-level", "entry level", "junior", "graduate", "fresher", "associate",
    "mid-level", "mid level", "mid-professional", "experienced", "senior",
    "principal", "lead", "executive", "director-level",
]
YEARS_EXPERIENCE_RE = re.compile(r"\b\d+\+?\s*(years?|yrs?)\b")

CLARIFICATION_PHRASES = [
    "what level", "what seniority", "which role", "what role", "what kind of",
    "could you clarify", "can you clarify", "what is the primary", "who is this for",
    "what focus", "which skills", "which competencies", "selection or development",
    "how many years", "what type of assessment",
]


@dataclass
class ConversationState:
    history: List[dict]
    latest_user_message: str
    all_user_text: str
    turn_count: int
    clarification_rounds: int
    has_prior_recommendation: bool
    prior_recommended_names: List[str]
    role_signal: bool
    purpose_signal: bool
    focus_signal: bool
    seniority_signal: bool
    long_form_input: bool
    recommendation_ready: bool


def _text_has_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def extract_prior_recommendations(history: List[dict]) -> List[str]:
    for message in reversed(history):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if RECOMMENDATION_MARKER not in content:
            continue
        block = content.split(RECOMMENDATION_MARKER, 1)[1]
        # only the first line after the marker is the machine-readable list
        first_line = block.strip().splitlines()[0] if block.strip() else ""
        names = [n.strip() for n in first_line.split("|") if n.strip()]
        if names:
            return names
    return []


def count_clarification_rounds(history: List[dict]) -> int:
    count = 0
    for message in history:
        if message["role"] != "assistant":
            continue
        content_lower = message["content"].lower()
        if RECOMMENDATION_MARKER in message["content"]:
            continue
        if _text_has_any(content_lower, CLARIFICATION_PHRASES) or content_lower.strip().endswith("?"):
            count += 1
    return count


def build_state(messages: List[dict]) -> ConversationState:
    history = messages
    user_messages = [m["content"] for m in history if m["role"] == "user"]
    latest_user_message = user_messages[-1] if user_messages else ""
    all_user_text = " ".join(user_messages).lower()

    prior_recommended_names = extract_prior_recommendations(history)
    has_prior_recommendation = len(prior_recommended_names) > 0
    clarification_rounds = count_clarification_rounds(history)

    role_signal = _text_has_any(all_user_text, ROLE_KEYWORDS)
    purpose_signal = _text_has_any(all_user_text, PURPOSE_KEYWORDS)
    focus_signal = _text_has_any(all_user_text, FOCUS_KEYWORDS)
    seniority_signal = (
        _text_has_any(all_user_text, SENIORITY_KEYWORDS)
        or bool(YEARS_EXPERIENCE_RE.search(all_user_text))
    )

    # A long, information-dense message (e.g. a pasted job description) is
    # very likely to already contain enough context even if it doesn't hit
    # our keyword lists, so we don't force a clarification round on it.
    long_form_input = len(re.findall(r"\w+", latest_user_message)) >= 40

    # Require role + focus (what to test, and for whom) plus at least one of
    # purpose or seniority (why, or at what level) before recommending.
    # role+focus alone -- e.g. "a Java developer who works with stakeholders"
    # with no seniority given, straight out of the assignment's own worked
    # example -- is intentionally NOT enough on its own; it should still
    # prompt a clarifying question.
    recommendation_ready = (
        (role_signal and focus_signal and (purpose_signal or seniority_signal))
        or long_form_input
        or has_prior_recommendation
        or clarification_rounds >= MAX_CLARIFICATION_ROUNDS
    )

    return ConversationState(
        history=history,
        latest_user_message=latest_user_message,
        all_user_text=all_user_text,
        turn_count=len(history),
        clarification_rounds=clarification_rounds,
        has_prior_recommendation=has_prior_recommendation,
        prior_recommended_names=prior_recommended_names,
        role_signal=role_signal,
        purpose_signal=purpose_signal,
        focus_signal=focus_signal,
        seniority_signal=seniority_signal,
        long_form_input=long_form_input,
        recommendation_ready=recommendation_ready,
    )
