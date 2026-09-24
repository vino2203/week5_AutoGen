# Backend: framework advisor for small customer-support projects

The user describes the project (free text, optional form fields) and which options they are considering
(RAG, n8n, CrewAI, Autogen). **Every agent is an LLM call**; there is no offline mode and no rule-based fallback.

## Setup

```bash
cd Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put your key in Backend/.env:   OPENAI_API_KEY=sk-...
```

Without a key the backend starts, but `/chat` answers `503: OPENAI_API_KEY is missing...` and `/status` says
`"llm": "not configured"`. Nothing is faked.

## Run

```bash
uvicorn agents.human_loop_agent:app --port 8710
python run_manager.py samples/description_only.json     # CLI
```

## How it works

| Agent | Role (all AI) |
|---|---|
| Manager (0) | An Autogen agent that runs the workflow itself by calling three tools: `set_project_brief` (turns your text and form fields into a structured brief, with an assumption for every guess), `analyze_framework` (once per option, all in the same step so they run in parallel) and `recommend_stack`. It never asks questions (only an empty description gets one). What the user set explicitly always wins over its brief, and it cannot widen the options you chose. |
| RAG / n8n / CrewAI / Autogen (1, 2, 3, 5) | Each estimates its framework used alone for this project: requirements covered/missing, setup hours and skill, daily cost, latency, recall (if documents), risks, a Mermaid design and a walkthrough. Each has a timeout and one retry; a failed one is excluded and shown as failed. |
| Recommender (6) | Builds candidate stacks (singles and sensible pairs), combines their numbers, flags hard-limit failures, scores five criteria 0-10, and writes the rationale, design, phased plan, upgrade triggers, risks and recall checklist. |

Code only does plumbing: it validates every model reply against a schema (an invalid tool call or JSON goes back to the
model with the error, once), refuses tool calls that break the rules (unknown option, brief not saved, duplicate
analysis), reminds the manager once if it stops before recommending (otherwise the session is an error, never a guess),
drops requirement keys a model invented, and **recomputes the weighted total from the
AI's criterion scores and your weights** (so the table adds up), then orders passing stacks first. A model failure is
reported as an error, never replaced by canned text.

Costs, latencies, hours and recall are **the model's estimates** from the assumptions each report lists. They are
planning figures, not benchmarks, and can differ between runs. Cost: about 3 manager turns plus one call per framework agent and one for the recommender per analysis (~8 calls with four options), and 1 call per re-score.

## API

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Start, approve/reject, or re-score a session |
| `GET /sessions/{id}` | Stored session |
| `GET /status` | AI ready or not, model settings, last log line per agent |
| `GET /health` | Liveness |

`POST /chat` body: `{"input": {...}}` to start (see `samples/`); `{"session_id", "message": "approve"|"reject"}`;
`{"session_id", "constraints": {...}, "weights": {...}}` to re-score (the recommender runs again, the framework agents do not).

## Config

`config.yaml`: model, temperature, max tokens, timeout, retries, default limits and weights.
`.env`: `OPENAI_API_KEY`, optional `LLM_MODEL`, `DATA_DIR` (logs and `agent_reports.db`).

## Tests

```bash
pytest   # uses a fake model, so it needs no key and checks the plumbing, not the quality of real AI output
```
