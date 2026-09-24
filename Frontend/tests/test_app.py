"""UI tests with Streamlit's AppTest; the backend is replaced by recorded real responses."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import api as api_module

FIX = Path(__file__).parent / "fixtures"
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def load(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def backend(monkeypatch):
    """Stub Api methods; records calls so tests can check payloads."""
    calls = []
    responses = {"chat": [load("analyzed.json")]}

    def chat(self, **body):
        calls.append(body)
        return responses["chat"].pop(0)

    monkeypatch.setattr(api_module.Api, "chat", chat)
    monkeypatch.setattr(api_module.Api, "status", lambda self: load("status.json"))
    return calls, responses


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception


def test_initial_screen_has_no_result(backend):
    at = run_app()
    assert len(at.chat_message) == 0
    assert not [b for b in at.button if b.label == "Approve"]  # nothing to approve yet
    assert next(b for b in at.button if b.label == "Re-score").disabled


def test_analyze_sends_form_and_shows_result(backend):
    calls, _ = backend
    at = run_app()
    at.text_area(key="f_description").set_value("WhatsApp FAQ bot").run()
    click(at, "Analyze")
    body = calls[0]
    assert body["input"]["project_description"] == "WhatsApp FAQ bot"
    assert body["input"]["requirements"]["channels"] == []  # left open: the AI reads them from the text
    assert body["input"]["requirements"]["document_upload"] is None
    assert "session_id" not in body
    assert len(at.chat_message) == 2  # your request + the recommendation
    assert any("Recommended" in s.value for s in at.subheader)
    assert [b for b in at.button if b.label == "Approve"]
    charts = at.get("graphviz_chart")  # the design diagrams are actually drawn (recommendation + agent reports)
    assert len(charts) >= 1 and all("digraph" in c.proto.spec for c in charts)


def test_approve_calls_backend_with_session(backend):
    calls, responses = backend
    responses["chat"].append(load("approved.json"))
    at = run_app()
    click(at, "Analyze")
    click(at, "Approve")
    assert calls[1] == {"session_id": load("analyzed.json")["session_id"], "message": "approve"}
    assert any("final recommendation" in m.markdown[0].value.lower() for m in at.chat_message if m.markdown)


def test_rescore_sends_thresholds_only(backend):
    calls, responses = backend
    responses["chat"].append(load("analyzed.json"))
    at = run_app()
    click(at, "Analyze")
    at.number_input(key="f_latency").set_value(3000).run()
    click(at, "Re-score")
    body = calls[1]
    assert body["session_id"] and "input" not in body
    assert body["constraints"]["max_latency_ms"] == 3000


def test_needs_input_shows_questions(backend):
    _, responses = backend
    responses["chat"][:] = [load("needs_input.json")]
    at = run_app()
    click(at, "Analyze")
    text = " ".join(m.markdown[0].value for m in at.chat_message if m.markdown)
    assert "Describe the project" in text


def test_backend_offline_is_reported(monkeypatch):
    def down(self, **kw):
        raise api_module.ApiError("Cannot reach the backend")

    monkeypatch.setattr(api_module.Api, "chat", down)
    monkeypatch.setattr(api_module.Api, "status", lambda self: down(self))
    at = run_app()
    click(at, "Analyze")
    assert any("Cannot reach the backend" in m.markdown[0].value for m in at.chat_message if m.markdown)


def test_chat_text_moves_forward_and_is_sent_as_the_description(backend):
    calls, _ = backend
    at = run_app()
    at.chat_input[0].set_value("gmail and sms").run()
    assert not at.exception
    body = calls[0]["input"]
    assert body["project_description"] == "gmail and sms"
    assert body["requirements"]["channels"] == []  # the manager AI reads them; the UI does not guess
    assert any("Recommended" in s.value for s in at.subheader)  # no questions in between


def test_free_text_after_a_result_refines_and_reanalyzes(backend):
    calls, responses = backend
    responses["chat"].append(load("analyzed.json"))
    at = run_app()
    at.text_area(key="f_description").set_value("support bot").run()
    click(at, "Analyze")
    at.chat_input[0].set_value("also on telegram, about 300 messages per day").run()
    assert not at.exception
    body = calls[1]
    assert body["session_id"] == load("analyzed.json")["session_id"]
    assert body["input"]["project_description"] == "support bot also on telegram, about 300 messages per day"


def test_explicit_form_choices_reach_the_backend(backend):
    calls, _ = backend
    at = run_app()
    at.text_area(key="f_description").set_value("support bot").run()
    at.selectbox(key="f_docs").select("No").run()
    at.number_input(key="f_msgs").set_value(30).run()
    click(at, "Analyze")
    r = calls[0]["input"]["requirements"]
    assert r["document_upload"] is False and r["messages_per_day"] == 30 and r["human_approval_of_replies"] is None


def test_what_the_ai_understood_is_shown(backend):
    at = run_app()
    click(at, "Analyze")
    labels = [e.label for e in at.expander]
    assert "What I understood from your input" in labels
    text = " ".join(m.value for m in at.markdown)
    assert "Channels:" in text and "gmail, sms" in text and "Read the channels from the description." in text


def test_missing_key_is_shown_as_a_banner_and_chat_error(monkeypatch):
    monkeypatch.setattr(api_module.Api, "status", lambda self: load("status_no_key.json"))

    def refuse(self, **kw):
        raise api_module.ApiError("OPENAI_API_KEY is missing. Put it in Backend/.env (copy .env.example) and restart the backend.")

    monkeypatch.setattr(api_module.Api, "chat", refuse)
    at = run_app()
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
    click(at, "Analyze")
    assert any("OPENAI_API_KEY" in m.markdown[0].value for m in at.chat_message if m.markdown)


def test_agent_status_only_shows_agents_used_in_the_current_analysis(backend):
    _, responses = backend
    resp = load("analyzed.json")
    resp["reports"] = [r for r in resp["reports"] if r["agent_id"] == "agent_3_crewai"]
    responses["chat"][:] = [resp]
    at = run_app()
    click(at, "Analyze")  # description empty in the UI, but the stub answers anyway
    rows = " ".join(m.value for m in at.sidebar.markdown)
    for name in ("RAG", "n8n", "Autogen"):
        assert f"<b>{name}</b> · <small>not used in this analysis</small>" in rows
    assert "<b>CrewAI</b> · <small>not used" not in rows
    assert "<b>Manager</b> · <small>not used" not in rows
