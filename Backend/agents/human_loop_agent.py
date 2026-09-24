"""The FastAPI app (/chat, /status) in front of the manager. Agent 6 lives in recommender.py.

Run:  uvicorn agents.human_loop_agent:app --port 8710
"""
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import storage
from .config import load_config
from .llm import KEY_HELP, LLMClient, LLMNotConfigured, get_llm
from .manager import Manager
from .schemas import Constraints, ProjectInput, Recommendation, Session, Weights


# ---------------------------------------------------------------- presentation
def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {s}" for s in items)


def render_markdown(rec: Recommendation) -> str:
    parts = [
        f"## {rec.headline}",
        rec.rationale,
        *(f"> Warning: {w}" for w in rec.warnings),
        "### Comparison\n" + rec.comparison_table_md,
        "### Simple design\n```mermaid\n" + rec.simple_design + "\n```",
        "### How it works\n" + _bullets(rec.how_it_works),
        "### Phased plan\n" + _bullets(rec.phased_plan),
        f"### Cost\n~${rec.estimated_daily_cost_usd}/day (~${rec.estimated_monthly_cost_usd}/month)",
        "### When to upgrade\n" + _bullets(rec.upgrade_triggers),
        "### Risks\n" + _bullets(rec.risks),
    ]
    if rec.recall_checklist:
        parts.append("### Recall validation checklist\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(rec.recall_checklist, 1)))
    parts.append("### Assumptions\n" + _bullets(rec.assumptions))
    return "\n\n".join(parts)


# ------------------------------------------------------------------------- API
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    input: Optional[ProjectInput] = None  # start a session (or re-run one) with the project
    message: Optional[str] = None  # "approve" or "reject"
    constraints: Optional[Constraints] = None  # re-score with changed limits
    weights: Optional[Weights] = None  # re-score with changed weights


class ChatResponse(Session):
    message: str = ""


def _message(s: Session) -> str:
    if s.status == "needs_input":
        return "\n".join(s.questions)
    if s.status == "error":
        return "Analysis failed: " + "; ".join(f"{e.agent_id}: {e.error}" for e in s.errors)
    body = render_markdown(s.recommendation) if s.recommendation else ""
    intro = {
        "awaiting_approval": "Here is my suggestion. Reply `approve`, or `reject` (optionally with changed constraints/weights to re-score).",
        "approved": "Approved. Final recommendation delivered.",
        "rejected": "Rejected. Send a new `input` (or changed constraints/weights) to try again.",
    }[s.status]
    failed = "".join(f"\n> Agent {e.agent_id} failed and was excluded: {e.error}" for e in s.errors)
    return f"{intro}{failed}\n\n{body}"


def create_app(llm: "LLMClient | None | Literal['auto']" = "auto") -> FastAPI:
    """`llm='auto'` reads OPENAI_API_KEY from Backend/.env. Pass None to run without a model (every /chat then fails clearly)."""
    problem: Optional[str] = None
    if llm == "auto":
        try:
            llm = get_llm()
        except LLMNotConfigured as e:
            llm, problem = None, str(e)
    elif llm is None:
        problem = KEY_HELP
    cfg = load_config()
    manager = Manager(llm, cfg) if llm is not None else None
    app = FastAPI(title="Framework advisor for small customer-support projects")

    def need_manager() -> Manager:
        if manager is None:
            raise HTTPException(503, problem or KEY_HELP)
        return manager

    def respond(session: Session) -> ChatResponse:
        return ChatResponse(**session.model_dump(), message=_message(session))

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/status")
    async def status():
        return {
            "llm": "ready" if llm is not None else "not configured",
            "problem": problem,
            "model": os.getenv("LLM_MODEL") or cfg["model"],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "agents": storage.last_events(),
        }

    @app.get("/sessions/{session_id}", response_model=Session)
    async def get_session(session_id: str):
        session = need_manager().load(session_id)
        if session is None:
            raise HTTPException(404, "unknown session_id")
        return session

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        m = need_manager()
        if req.input is not None:
            return respond(await m.start(req.input, req.session_id))
        if not req.session_id:
            raise HTTPException(400, "Send `input` to start a session.")
        session = m.load(req.session_id)
        if session is None:
            raise HTTPException(404, "unknown session_id")
        if session.status == "needs_input":
            raise HTTPException(400, "This session is waiting for a description: send `input` again.")
        decision = (req.message or "").strip().lower()
        if req.constraints or req.weights:
            try:
                return respond(await m.revise(session, req.constraints, req.weights))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                raise HTTPException(502, f"The recommender could not re-score: {e}")
        if decision in ("approve", "approved", "yes", "ok"):
            if session.status != "awaiting_approval":
                raise HTTPException(409, f"Nothing to approve: session is '{session.status}'.")
            return respond(m.decide(session, approve=True))
        if decision in ("reject", "rejected", "no"):
            if session.status != "awaiting_approval":
                raise HTTPException(409, f"Nothing to reject: session is '{session.status}'.")
            return respond(m.decide(session, approve=False))
        raise HTTPException(400, "Send `message`: approve | reject, or new `constraints` / `weights`.")

    return app


app = create_app()
