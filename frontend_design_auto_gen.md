# Frontend UI/UX Design: Framework Advisor

Streamlit front end (`Frontend/`) for the FastAPI backend in `Backend/`. It replaces the earlier draft, which
assumed a free-text chatbot with document ingestion; the backend is a structured advisor, so the UI follows that.

## What the user can do
- Describe a project with a form (options to compare, channels, documents yes/no, messages per day, approval, skill, time).
- Set thresholds (recall, latency, budget) and criteria weights, and **re-score** the existing analysis without re-running agents.
- Read the result: summary, comparison table, Mermaid design + how-it-works, phased plan, risks, recall checklist, per-agent reports.
- **Approve** or **reject** the suggestion, in the chat box or with buttons.
- See each agent's last log line and status, and the read-only model settings.
- Export the conversation as JSON or Markdown.

## Decisions (agreed)
| Earlier draft | Now |
|---|---|
| File uploader (PDF/DOCX/TXT) into RAG | Toggle "answers come from documents" plus optional count/size; the advisor does not ingest documents |
| Model / temperature / max tokens / system prompt per user | Shown read-only; set in backend `config.yaml` / `.env` (OpenAI only) |
| Streaming, memory reset, hidden `end` token | Dropped: the backend has none of these |
| Live heartbeat | Last log line and status per agent |
| POST `{message, upload_id, settings}` | POST `{session_id, input, message, constraints, weights}` |
| Custom HTML bubbles | `st.chat_message` plus CSS for colours |

## Layout
1. **Sidebar:** project form, thresholds, weights, Analyze / Re-score, agent status, model settings, export, dark mode, advanced (backend URL, timeout, raw JSON debug).
2. **Main:** chat timeline; the latest recommendation is shown as tabs, earlier ones are collapsed.
3. **Footer:** Approve / Reject / Show comparison buttons and a chat box (a description starts an analysis; `approve` / `reject` decide).

## Theme
Teal `#00d1b2`, navy `#001f3f`, light grey `#f5f5f5`, Roboto; fade-in messages; dark mode via CSS variables; visible keyboard focus.

## Files
```
Frontend/
  app.py       layout and flow
  api.py       httpx client (friendly errors)
  logic.py     form -> payload, exports (pure Python)
  render.py    result tabs, Mermaid embed, theme CSS
  styles.css
  tests/       logic tests + AppTest UI tests with recorded backend responses
```
Docker is not included.
