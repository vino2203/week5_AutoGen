"""Agent 0: the manager, an LLM that runs the whole workflow by calling tools.

It reads the user's input and decides for itself what to do: it saves a structured brief, calls the framework agents it
needs (in parallel), then asks the recommender for the final answer. The tools below are the only things it can do.
The code around it only enforces that the tools are used sensibly and reports what happened.
"""
import asyncio
import json
import logging
import uuid
from typing import Annotated, Callable, Literal, Optional

from pydantic import ValidationError

from . import storage
from .autogen_agent import AutogenAgent
from .config import load_config
from .crewai_agent import CrewAIAgent
from .knowledge_base import KnowledgeAgent
from .llm import LLMClient
from .n8n_agent import N8nAgent
from .rag_agent import RagAgent
from .recommender import RecommenderAgent
from .schemas import (
    AgentError,
    Brief,
    Constraints,
    FrameworkReport,
    ProjectInput,
    Recommendation,
    Session,
    Weights,
)

log = logging.getLogger(__name__)

AGENTS: dict[str, type[KnowledgeAgent]] = {
    "rag": RagAgent,
    "n8n": N8nAgent,
    "crewai": CrewAIAgent,
    "autogen": AutogenAgent,
}

SYSTEM = """You are Agent 0, the manager of a framework-selection workflow for SMALL or MICRO customer-support projects.
You do not answer the user's question yourself. You run the workflow by calling your tools, in this order:

1. set_project_brief (once). Turn the user's free text and form fields into a complete structured brief, so the
   specialists can start at once. You never ask the user questions: infer what the text implies, make sensible modest
   assumptions for the rest, and record an assumption for every guess.
2. analyze_framework, once for EACH option in the brief. Call them all in the same step so they run in parallel.
3. recommend_stack (once), after every analysis has come back (a failed analysis is fine; do not retry it).
4. Finish with one short sentence saying what was recommended.

Rules for the brief:
- A value the user gave explicitly (non-null, non-empty) is final: copy it. Fill the rest from the description.
- options_to_compare: only the options the user named (in the field or the text); if none were named, all four.
- channels: lowercase names such as gmail, whatsapp, telegram, slack, sms, email, web chat. If none are stated or implied,
  use ["web chat"] and say so in the assumptions.
- document_upload: true if the answers should come from the owner's documents / FAQs / knowledge base.
- messages_per_day: use a stated number; otherwise assume a modest figure for a small business and say so.
- human_approval_of_replies / multi_step_reasoning: true only if the text asks for it; otherwise false.
- limits: copy explicit values, otherwise use the defaults you are given. If "time_available" text is given (for example
  "1 week"), convert it to hours (1 day = 8 h, 1 week = 40 h).
- Do not invent facts about the business."""


class _Run:
    """What the tools have produced so far in one analysis."""

    def __init__(self, given: ProjectInput):
        self.given = given
        self.project: Optional[ProjectInput] = None
        self.assumptions: list[str] = []
        self.reports: dict[str, FrameworkReport] = {}
        self.errors: dict[str, AgentError] = {}
        self.started: set[str] = set()
        self.recommendation: Optional[Recommendation] = None
        self.recommender_error: Optional[str] = None

    @property
    def options(self) -> list[str]:
        return self.project.options_to_compare if self.project else []

    @property
    def unhandled(self) -> list[str]:
        return [o for o in self.options if o not in self.reports and o not in self.errors]

    @property
    def finished(self) -> bool:
        return self.recommendation is not None


class Manager:
    agent_id = "agent_0_manager"

    def __init__(self, llm: LLMClient, cfg: Optional[dict] = None):
        self.llm = llm
        self.cfg = cfg or load_config()
        self.recommender = RecommenderAgent(llm)

    # ------------------------------------------------------------------ session
    def load(self, session_id: str) -> Optional[Session]:
        raw = storage.load_run(session_id)
        return Session.model_validate_json(raw) if raw else None

    def _save(self, session: Session) -> Session:
        storage.save_run(session.session_id, session.status, session.model_dump_json())
        return session

    async def start(self, project_input: ProjectInput, session_id: Optional[str] = None) -> Session:
        """New evaluation, run by the manager AI."""
        session_id = session_id or uuid.uuid4().hex[:12]
        previous = self.load(session_id)
        revision = previous.revision + 1 if previous else 0

        if not project_input.project_description.strip():
            storage.log_event(self.agent_id, "completed", "needs a description")
            return self._save(Session(
                session_id=session_id, status="needs_input", revision=revision, project=project_input,
                questions=["Describe the project in a sentence or two, for example: answer customers on WhatsApp from our FAQ."],
            ))

        run = _Run(project_input)
        storage.log_event(self.agent_id, "running", "planning the analysis")
        try:
            await self._drive(run)
        except Exception as e:  # e.g. an invalid API key or a network error
            storage.log_event(self.agent_id, "error", str(e))
            return self._save(Session(
                session_id=session_id, status="error", revision=revision, project=run.project or project_input,
                reports=list(run.reports.values()), errors=[AgentError(agent_id=self.agent_id, error=str(e))],
            ))

        errors = list(run.errors.values())
        if run.project is None:
            errors.append(AgentError(agent_id=self.agent_id, error="the manager did not save a project brief"))
        elif not run.reports:
            errors.append(AgentError(agent_id=self.agent_id, error="no framework analysis succeeded"))
        elif run.recommender_error and not run.finished:
            errors.append(AgentError(agent_id=self.recommender.agent_id, error=run.recommender_error))
        elif not run.finished:
            errors.append(AgentError(agent_id=self.agent_id, error="the manager did not ask for a recommendation"))
        for option in run.unhandled:
            errors.append(AgentError(agent_id=AGENTS[option].agent_id, error="the manager did not run this analysis"))

        session = Session(
            session_id=session_id, status="awaiting_approval" if run.finished else "error", revision=revision,
            project=run.project or project_input, assumptions=run.assumptions,
            reports=[run.reports[o] for o in run.options if o in run.reports],
            errors=errors, recommendation=run.recommendation,
        )
        storage.log_event(self.agent_id, "completed" if run.finished else "error", f"session {session_id} {session.status}")
        return self._save(session)

    async def revise(self, session: Session, constraints: Optional[Constraints], weights: Optional[Weights]) -> Session:
        """Ask the recommender to re-score the existing reports with changed limits or weights."""
        if session.project is None or not session.reports:
            raise ValueError("session has no analysis to re-score")
        project = session.project.model_copy(deep=True)
        if constraints:
            for k, v in constraints.model_dump(exclude_none=True, exclude={"time_available"}).items():
                setattr(project.constraints, k, v)
        if weights:
            project.weights = weights.resolved(project.weights.model_dump())
        session.project = project
        session.revision += 1
        session.status = "awaiting_approval"
        session.recommendation = await self.recommender.recommend(project, session.reports, session.assumptions)
        storage.log_event(self.agent_id, "completed", f"session {session.session_id} re-scored (revision {session.revision})")
        return self._save(session)

    def decide(self, session: Session, approve: bool) -> Session:
        session.status = "approved" if approve else "rejected"
        storage.log_event(self.agent_id, "completed", f"session {session.session_id} {session.status}")
        return self._save(session)

    # ------------------------------------------------------------ the AI manager
    async def _drive(self, run: _Run) -> None:
        """Let the manager AI work. If it stops early, remind it once of what is left."""
        tools = self._tools(run)
        for attempt in range(2):
            await asyncio.wait_for(
                self.llm.run_with_tools(self.agent_id, SYSTEM, self._task(run), tools), self.cfg["manager_timeout_seconds"]
            )
            if run.finished:
                return
            log.warning("manager stopped early (attempt %d): brief=%s left=%s", attempt + 1, bool(run.project), run.unhandled)

    def _task(self, run: _Run) -> str:
        task = (
            "USER INPUT (JSON; null or empty means the user left it open):\n"
            + json.dumps(run.given.model_dump(exclude={"weights"}), indent=2)
            + f"\n\nDEFAULT LIMITS: {json.dumps(self.cfg['default_constraints'])}"
        )
        if run.project is None and not run.reports:
            return task
        done = ", ".join(run.reports) or "none"
        failed = ", ".join(run.errors) or "none"
        return (
            f"{task}\n\nPROGRESS SO FAR: the brief is {'saved' if run.project else 'NOT saved yet'}; analyses done: {done}; "
            f"failed: {failed}; still to analyze: {', '.join(run.unhandled) or 'none'}; recommendation: "
            f"{'done' if run.finished else 'NOT done yet'}. Continue the workflow from where it stopped."
        )

    def _tools(self, run: _Run) -> list[Callable]:
        cfg = self.cfg
        dc = cfg["default_constraints"]

        async def set_project_brief(
            options_to_compare: Annotated[list[str], "Subset of rag, n8n, crewai, autogen: only those the user named, else all four"],
            channels: Annotated[list[str], "Lowercase channel names; ['web chat'] if none stated or implied"],
            document_upload: Annotated[bool, "True if answers should come from the owner's documents / FAQs"],
            messages_per_day: Annotated[int, "Stated number, otherwise a modest assumed figure"],
            human_approval_of_replies: Annotated[bool, "True only if the text asks for a person to approve replies"],
            multi_step_reasoning: Annotated[bool, "True only if the text asks for several cooperating agents / multi-step reasoning"],
            budget_per_day_usd: Annotated[float, "Daily budget limit"],
            max_latency_ms: Annotated[int, "Maximum acceptable reply time in milliseconds"],
            min_recall: Annotated[float, "Minimum document-retrieval recall, 0 to 1"],
            user_skill: Annotated[Literal["no-code", "low-code", "code"], "The user's technical skill"],
            time_available_hours: Annotated[float, "Hours available for setup"],
            assumptions: Annotated[list[str], "One short sentence for every value you had to guess or infer"],
        ) -> str:
            """Save the structured project brief. Call this exactly once, first."""
            if run.project is not None:
                return "The brief is already saved. Continue with analyze_framework."
            try:
                brief = Brief(
                    options_to_compare=options_to_compare, channels=channels, document_upload=document_upload,
                    messages_per_day=messages_per_day, human_approval_of_replies=human_approval_of_replies,
                    multi_step_reasoning=multi_step_reasoning, assumptions=assumptions,
                    constraints=Constraints(
                        budget_per_day_usd=budget_per_day_usd, max_latency_ms=max_latency_ms, min_recall=min_recall,
                        user_skill=user_skill, time_available_hours=time_available_hours,
                    ),
                )
            except (ValidationError, ValueError) as e:
                return f"INVALID brief: {e}. Call set_project_brief again with corrected values."
            run.project, run.assumptions = self._merge(run.given, brief, dc)
            storage.log_event(self.agent_id, "running", f"brief saved; options: {', '.join(run.options)}")
            return f"Brief saved. Now call analyze_framework once for each of: {', '.join(run.options)} (all in the same step)."

        async def analyze_framework(
            option: Annotated[Literal["rag", "n8n", "crewai", "autogen"], "The framework to analyze"],
        ) -> str:
            """Ask one framework specialist to analyze its framework for this project. Call once per option."""
            if run.project is None:
                return "Save the brief first with set_project_brief."
            if option not in run.options:
                return f"'{option}' is not one of the options to compare ({', '.join(run.options)}). Skipped."
            if option in run.started:
                return f"'{option}' was already requested. Do not call it again."
            run.started.add(option)
            try:
                report = await self._run_agent(option, run.project)
            except Exception as e:
                run.errors[option] = AgentError(agent_id=AGENTS[option].agent_id, error=str(e) or type(e).__name__)
                return f"FAILED: {option} could not be analyzed ({e}). Do not retry it; carry on without it."
            run.reports[option] = report
            covered, missing = report.fit_to_requirements.covered, report.fit_to_requirements.missing
            return (
                f"{report.framework}: covers {len(covered)} of {len(covered) + len(missing)} requirements, "
                f"~${report.estimated_daily_cost_usd}/day, ~{report.estimated_latency_ms} ms, ~{report.setup_effort.hours:g} h setup."
            )

        async def recommend_stack() -> str:
            """Ask the recommender to compare the finished analyses and pick the best stack. Call once, last."""
            if run.project is None or not run.reports:
                return "Nothing to recommend yet: save the brief and analyze at least one framework first."
            running = [o for o in run.started if o not in run.reports and o not in run.errors]
            if running:
                return f"Wait: the analysis of {', '.join(running)} is still running."
            if run.finished:
                return "The recommendation is already done."
            try:
                run.recommendation = await self.recommender.recommend(run.project, list(run.reports.values()), run.assumptions)
            except Exception as e:
                run.recommender_error = str(e) or type(e).__name__
                return f"FAILED: the recommender could not finish ({e})."
            return f"Done. {run.recommendation.headline}."

        return [set_project_brief, analyze_framework, recommend_stack]

    def _merge(self, given: ProjectInput, brief: Brief, dc: dict) -> tuple[ProjectInput, list[str]]:
        """The manager's brief, with everything the user set explicitly taking priority."""
        project = given.model_copy(deep=True)
        r, gr = project.requirements, given.requirements
        project.options_to_compare = given.options_to_compare or brief.options_to_compare
        r.channels = gr.channels or brief.channels
        r.document_upload = gr.document_upload if gr.document_upload is not None else brief.document_upload
        r.messages_per_day = gr.messages_per_day if gr.messages_per_day is not None else brief.messages_per_day
        r.human_approval_of_replies = (
            gr.human_approval_of_replies if gr.human_approval_of_replies is not None else brief.human_approval_of_replies
        )
        r.multi_step_reasoning = gr.multi_step_reasoning if gr.multi_step_reasoning is not None else brief.multi_step_reasoning

        c, given_c, ai_c = project.constraints, given.constraints, brief.constraints
        for field in ("budget_per_day_usd", "max_latency_ms", "min_recall", "user_skill", "time_available_hours"):
            explicit, inferred = getattr(given_c, field), getattr(ai_c, field)
            setattr(c, field, explicit if explicit is not None else inferred if inferred is not None else dc[field])

        assumptions = list(brief.assumptions)
        if given.weights is None:
            assumptions.append("Criteria weights not given; the default weights were used.")
        project.weights = (given.weights or Weights()).resolved(self.cfg["default_weights"])
        return project, assumptions

    # ------------------------------------------------------- framework agents
    async def _run_agent(self, key: str, project: ProjectInput) -> FrameworkReport:
        agent = AGENTS[key]()
        timeout = self.cfg["timeout_seconds"]
        last: Exception = RuntimeError("not run")
        for _ in range(self.cfg.get("retries", 1) + 1):
            try:
                return await asyncio.wait_for(agent.analyze(project, self.llm), timeout)
            except Exception as e:
                last = e if str(e) else TimeoutError(f"timed out after {timeout}s")
                log.warning("%s failed: %s", agent.agent_id, last)
        raise last
