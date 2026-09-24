import json

import pytest
from fastapi.testclient import TestClient

from agents.human_loop_agent import create_app
from tests.conftest import SAMPLES
from tests.fakes import FakeLLM


@pytest.fixture
def fake():
    return FakeLLM()


@pytest.fixture
def client(fake):
    return TestClient(create_app(fake))


@pytest.fixture
def payload():
    return json.loads((SAMPLES / "whatsapp_gmail_faq.json").read_text())


def test_without_a_key_everything_fails_clearly_and_nothing_is_faked():
    c = TestClient(create_app(None))
    assert c.get("/health").json() == {"ok": True}
    status = c.get("/status").json()
    assert status["llm"] == "not configured" and "OPENAI_API_KEY" in status["problem"]
    r = c.post("/chat", json={"input": {"project_description": "a bot"}})
    assert r.status_code == 503 and "Backend/.env" in r.json()["detail"]


def test_auto_mode_reads_the_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = TestClient(create_app("auto"))
    assert c.get("/status").json()["llm"] == "not configured"


def test_status_reports_the_model(client):
    s = client.get("/status").json()
    assert s["llm"] == "ready" and s["model"] and "temperature" in s and "max_tokens" in s


def test_full_flow_start_rescore_approve(client, payload, fake):
    r = client.post("/chat", json={"input": payload}).json()
    assert r["status"] == "awaiting_approval" and "Recommended" in r["message"]
    sid = r["session_id"]

    r = client.post("/chat", json={"session_id": sid, "message": "reject", "constraints": {"max_latency_ms": 3000}}).json()
    assert r["status"] == "awaiting_approval" and r["revision"] == 1
    assert fake.calls.count("agent_6_recommender") == 2 and fake.calls.count("agent_1_rag") == 1

    r = client.post("/chat", json={"session_id": sid, "message": "approve"}).json()
    assert r["status"] == "approved"
    assert client.get(f"/sessions/{sid}").json()["status"] == "approved"
    assert client.get("/status").json()["agents"]["agent_0_manager"]["status"] == "completed"


def test_reject_without_changes(client, payload):
    sid = client.post("/chat", json={"input": payload}).json()["session_id"]
    assert client.post("/chat", json={"session_id": sid, "message": "reject"}).json()["status"] == "rejected"
    assert client.post("/chat", json={"session_id": sid, "message": "approve"}).status_code == 409


def test_description_alone_is_enough(client):
    r = client.post("/chat", json={"input": {"project_description": "Order status by email and SMS, 50 a day"}}).json()
    assert r["status"] == "awaiting_approval" and r["assumptions"]
    assert r["project"]["requirements"]["channels"] == ["gmail", "sms"]


def test_empty_description_asks_for_one(client):
    r = client.post("/chat", json={"input": {}}).json()
    assert r["status"] == "needs_input" and len(r["questions"]) == 1
    sid = r["session_id"]
    assert client.post("/chat", json={"session_id": sid, "message": "approve"}).status_code == 400
    r = client.post("/chat", json={"session_id": sid, "input": {"project_description": "a bot"}}).json()
    assert r["status"] == "awaiting_approval"


def test_a_manager_failure_is_shown_not_hidden(payload):
    c = TestClient(create_app(FakeLLM(scripts=[["I cannot help."], ["Still cannot."]])))
    r = c.post("/chat", json={"input": payload}).json()
    assert r["status"] == "error" and "agent_0_manager" in r["message"]


def test_rescore_failure_is_a_502(fake, payload):
    c = TestClient(create_app(fake))
    sid = c.post("/chat", json={"input": payload}).json()["session_id"]
    fake.fail_for.add("agent_6_recommender")
    r = c.post("/chat", json={"session_id": sid, "constraints": {"max_latency_ms": 500}})
    assert r.status_code == 502 and "could not re-score" in r.json()["detail"]


def test_bad_requests(client):
    assert client.post("/chat", json={"message": "approve"}).status_code == 400
    assert client.post("/chat", json={"session_id": "nope", "message": "approve"}).status_code == 404
    assert client.post("/chat", json={"input": {"options_to_compare": ["bogus"]}}).status_code == 422
    assert client.get("/sessions/nope").status_code == 404
