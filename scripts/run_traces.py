"""Runs the self-authored conversation traces end to end against the real
Groq-backed pipeline (in-process, no server needed) and checks:
  - every turn's classified query_type matches the trace's expectation
  - every response validates against the ChatResponse schema
  - the final recommendation count respects the trace's min/max assertions

Requires GROQ_API_KEY to be set (these traces exercise the real model, not
a mock -- the structural/gating logic already has hermetic coverage in
tests/test_response.py, which runs without any API key).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import GROQ_API_KEY
from app.schemas import ChatResponse
from app.services.classifier import classify
from app.services.response import handle_chat
from app.services.state import build_state

TRACE_DIR = Path(__file__).resolve().parent.parent / "tests" / "conversation_traces"


def run_trace(path: Path):
    trace = json.loads(path.read_text(encoding="utf-8"))
    assertions = trace["assertions"]
    expected_types = assertions.get("expect_query_type_by_turn", [])

    messages = []
    results = []
    problems = []

    for i, turn in enumerate(trace["turns"]):
        messages.append({"role": "user", "content": turn["user"]})

        state = build_state(messages)
        qtype = classify(state)
        if i < len(expected_types) and qtype != expected_types[i]:
            problems.append(f"turn {i + 1}: expected query_type={expected_types[i]!r}, got {qtype!r}")

        response = handle_chat(messages)
        try:
            ChatResponse(**response)
        except Exception as e:
            problems.append(f"turn {i + 1}: schema validation failed: {e}")

        messages.append({"role": "assistant", "content": response["reply"]})
        results.append(
            {
                "turn": i + 1,
                "query_type": qtype,
                "recommendations": len(response["recommendations"]),
                "end_of_conversation": response["end_of_conversation"],
            }
        )

    final_count = results[-1]["recommendations"] if results else 0
    min_rec = assertions.get("min_recommendations_by_end")
    if min_rec is not None and final_count < min_rec:
        problems.append(f"expected >= {min_rec} recommendations by end, got {final_count}")
    max_rec = assertions.get("max_recommendations_by_end")
    if max_rec is not None and final_count > max_rec:
        problems.append(f"expected <= {max_rec} recommendations by end, got {final_count}")

    return trace["id"], len(problems) == 0, problems, results


def main() -> None:
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set -- skipping live-LLM trace evaluation.")
        print("Structural/gating behavior already has hermetic coverage in tests/test_response.py.")
        print("Set GROQ_API_KEY and re-run this script for a full end-to-end evaluation.")
        return

    trace_files = sorted(TRACE_DIR.glob("*.json"))
    passed = 0
    for path in trace_files:
        trace_id, ok, problems, results = run_trace(path)
        print(f"[{'PASS' if ok else 'FAIL'}] {trace_id}")
        for r in results:
            print(
                f"    turn {r['turn']}: type={r['query_type']} "
                f"recs={r['recommendations']} end={r['end_of_conversation']}"
            )
        for p in problems:
            print(f"    ! {p}")
        passed += ok

    print(f"\n{passed}/{len(trace_files)} traces passed")


if __name__ == "__main__":
    main()
