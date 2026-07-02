# SHL Assessment Recommender

A conversational agent that takes a recruiter from a vague hiring need to a
grounded shortlist of SHL assessments through dialogue -- clarifying when
needed, refining on request, comparing named assessments, and refusing
anything outside SHL's assessment catalog.

Built for the SHL Labs take-home assignment. See `APPROACH.md` for design
rationale, trade-offs, and what didn't work.

## API

| Method | Path      | Purpose                                          |
|--------|-----------|---------------------------------------------------|
| GET    | `/health` | Readiness check -> `{"status": "ok"}`             |
| POST   | `/chat`   | Stateless conversational recommendation endpoint  |

`POST /chat` request/response shapes match the assignment spec exactly:

```json
// request
{"messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]}

// response
{
  "reply": "Got it -- what's the seniority level for this role?",
  "recommendations": [],
  "end_of_conversation": false
}
```

`recommendations` is only ever non-empty on a turn where the agent commits
to a shortlist (1-10 items, each with `name`, `url`, `test_type`).

## Project layout

```
app/
  main.py               FastAPI app: /health, /chat
  schemas.py            Pydantic request/response models (exact API contract)
  config.py             env vars, tunable constants
  services/
    catalog.py           loads data/processed/catalog.json
    retrieval.py          BM25 lexical search + synonym expansion
    state.py               stateless conversation-state builder
    classifier.py            rule-based intent classifier
    llm.py                    Groq client + prompt builders (JSON-mode output)
    response.py                orchestrates state -> retrieval -> LLM -> gated response
scripts/
  scrape_catalog.py     builds data/processed/catalog.json from the SHL catalog
  run_traces.py         runs tests/conversation_traces/*.json against the live pipeline
data/
  raw/                  scraped listing rows + per-assessment detail cache
  processed/catalog.json final catalog the app loads at runtime
tests/
  test_classifier.py, test_state.py, test_retrieval.py, test_response.py  (no API key needed)
  conversation_traces/  10 self-authored persona traces covering all 4 behaviors + probes
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=gsk_your_key  # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."

uvicorn app.main:app --reload
```

Rebuilding the catalog (only needed if you want to re-scrape):

```bash
python scripts/scrape_catalog.py
```

Running tests (no API key required -- the LLM is mocked):

```bash
pip install pytest
python -m pytest tests/ -v
```

Running the live conversation traces against the real model (needs `GROQ_API_KEY`):

```bash
python scripts/run_traces.py
```

## Deployment

Includes both a `Procfile` (Render/Railway/Heroku-style) and a `Dockerfile`.
Either way, set `GROQ_API_KEY` as an environment variable on the host. The
service reads `data/processed/catalog.json` at startup, so make sure that
file is included in the deployed artifact (it's committed, not gitignored --
only the raw per-assessment scrape cache under `data/raw/catalog_detail/`
is excluded).
