"""Base class for the knowledge agents (1, 2, 3, 5). Each one is an LLM expert on one framework."""
import json

from . import storage
from .llm import LLMClient, ask_json, clean_mermaid
from .schemas import ALL_OPTIONS, FitInfo, FrameworkReport, ProjectInput, ReportBody, required_keys

REPORT_FORMAT = """Reply with ONLY one JSON object, no other text:
{
  "summary": "2-3 plain-language sentences: what it is and how well it fits THIS project",
  "fit_to_requirements": {"covered": ["<requirement key>"], "missing": ["<requirement key>"]},
  "setup_effort": {"hours": 0, "skill_required": "no-code | low-code | code",
                   "steps": [{"step": "what to do", "hours": 0}]},
  "estimated_daily_cost_usd": 0.0,
  "estimated_latency_ms": 0,
  "long_term_score": 0,
  "expected_recall": null,
  "risks": ["..."],
  "works_well_with": ["rag | n8n | crewai | autogen"],
  "simple_design": "flowchart LR\\n    A[\\"...\\"] --> B[\\"...\\"]",
  "how_it_works": ["1) ...", "2) ..."],
  "project_specific_notes": ["..."],
  "assumptions": ["..."]
}

Rules:
- Estimate the framework used ALONE for this project. Every requirement key listed below must appear in exactly one of
  "covered" (the framework itself provides it, or you count the custom code needed in setup hours and set skill_required
  to "code") or "missing" (it cannot do this by itself and no such code is counted).
- estimated_daily_cost_usd = hosting per day + messages_per_day x LLM calls per message x (input tokens x input price
  + output tokens x output price). Unless you say otherwise use gpt-4o-mini prices ($0.15 in / $0.60 out per 1M tokens),
  about 1,500 input + 250 output tokens per call, and self-hosting on a small server (about $5/month, roughly $0.17/day;
  a managed cloud plan costs more). Frameworks that make several LLM calls per message cost proportionally more. Show
  the price, tokens, calls per message and hosting cost you used in "assumptions". Do not round every framework to the
  same number: work it out for this one.
- estimated_latency_ms: typical end-to-end time to produce one reply. One LLM call takes roughly 700-2000 ms; retrieval
  adds 50-300 ms; a workflow tool adds about 100 ms; a framework that makes N LLM calls one after another takes about N
  times a single call. Never quote less than the time of the LLM calls it needs.
- setup_effort.hours: realistic total for someone with the stated skill level; the steps' hours must add up to it.
  skill_required is what the work really needs: writing Python code means "code", even if the user is low-code.
- long_term_score: 0-10 for maintainability and extensibility (scalability is not required).
- expected_recall: 0-1 estimate of how often the right document passage is retrieved, ONLY if document retrieval is
  involved for this project, else null. It is an estimate to be validated, not a measurement.
- simple_design: a Mermaid flowchart ("flowchart LR", 5-8 nodes, every label in double quotes, plain --> arrows)
  built from THIS project's channels, documents, volume and approval step. No generic diagrams.
- how_it_works: 4-7 strings walking ONE real customer request through the design, naming concrete tools at each step.
- project_specific_notes: 2-4 things to configure for this project.
- If something about the project is unknown, state your assumption instead of guessing silently. Be honest about weaknesses."""


class KnowledgeAgent:
    agent_id: str
    key: str  # rag | n8n | crewai | autogen
    framework: str
    persona: str  # the system prompt

    @property
    def system_prompt(self) -> str:
        return (
            f"{self.persona} You advise the owner of a SMALL or MICRO customer-support project who may not be technical. "
            "Use plain language and honest, realistic numbers."
        )

    async def analyze(self, project: ProjectInput, llm: LLMClient) -> FrameworkReport:
        storage.log_event(self.agent_id, "running", f"analyzing {self.key}")
        try:
            body = await ask_json(llm, ReportBody, self.agent_id, self.system_prompt, self._user_prompt(project))
            report = self._finish(body, project)
        except Exception as e:
            storage.log_event(self.agent_id, "error", str(e))
            raise
        storage.log_event(self.agent_id, "completed", f"{self.key} report")
        return report

    def _finish(self, body: ReportBody, project: ProjectInput) -> FrameworkReport:
        """Keep the AI's content; only make its keys consistent so later stages can rely on them."""
        keys = required_keys(project)
        covered = [k for k in body.fit_to_requirements.covered if k in keys]
        missing = [k for k in keys if k not in covered]  # anything the AI did not claim is treated as not covered
        data = body.model_dump()
        data["fit_to_requirements"] = FitInfo(covered=covered, missing=missing).model_dump()
        data["works_well_with"] = [o for o in data["works_well_with"] if o in ALL_OPTIONS and o != self.key]
        data["simple_design"] = clean_mermaid(data["simple_design"])
        return FrameworkReport(agent_id=self.agent_id, framework=self.framework, **data)

    def _user_prompt(self, project: ProjectInput) -> str:
        r, c = project.requirements, project.constraints
        brief = {
            "project_description": project.project_description,
            "channels": r.channels,
            "answers_come_from_uploaded_documents": r.document_upload,
            "messages_per_day": r.messages_per_day,
            "human_approves_replies": r.human_approval_of_replies,
            "needs_multi_step_reasoning": r.multi_step_reasoning,
            "user_skill": c.user_skill,
            "time_available_hours": c.time_available_hours,
        }
        return (
            f"Analyze {self.framework} for this project.\n\nPROJECT (JSON):\n{json.dumps(brief, indent=2)}\n\n"
            f"REQUIREMENT KEYS (use exactly these strings in fit_to_requirements): {json.dumps(required_keys(project))}\n\n"
            f"{REPORT_FORMAT}"
        )
