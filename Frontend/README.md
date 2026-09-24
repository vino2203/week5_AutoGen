# Frontend: Streamlit UI for the framework advisor

Talks to the backend in [../Backend](../Backend) over HTTP (`/chat`, `/status`). Design notes: [../frontend_design_auto_gen.md](../frontend_design_auto_gen.md).

```bash
cd Frontend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# terminal 1: backend
cd ../Backend && uvicorn agents.human_loop_agent:app --port 8710   # needs OPENAI_API_KEY in Backend/.env
# terminal 2: UI (BACKEND_URL defaults to http://127.0.0.1:8710)
streamlit run app.py   # serves on http://127.0.0.1:8720 (see .streamlit/config.toml)
```

## Using it
1. Type a description in the chat box (or in the project card) and press Enter / **Analyze**. Everything else is optional: the AI reads it from your text and lists what it assumed under *What I understood*.
2. Read the result tabs: Summary, Comparison, Design, Plan, Risks, Recall check, Agent reports.
3. Change thresholds or weights and press **Re-score**, or **Approve** / **Reject**.
4. Export the conversation as JSON or Markdown from the sidebar.

## Notes
- Fields left on "Let the advisor decide" are sent as open; anything you set explicitly wins over the AI's reading. Free text after a recommendation refines the project and re-runs it.
- If the backend has no OpenAI key, a red banner says so.
- No document upload: the toggle only tells the advisor whether retrieval is needed (optional count and size are added to the description). The advisor recommends a stack; it does not ingest documents.
- Model, temperature and max tokens are shown read-only; they come from the backend's `config.yaml` / `.env`.
- Agent status is the last log line per agent, not a live heartbeat.
- Diagrams load mermaid.js from a CDN; the source is shown under each diagram if it cannot load.
- Dark mode is CSS-only. Bubble colours rely on Streamlit's current chat markup and may need tweaks after upgrades.

## Tests
```bash
pytest   # test_logic.py is pure Python; test_app.py drives the UI with Streamlit's AppTest and recorded backend responses
```
