"""Agent 6: the recommender. An LLM compares the framework reports and recommends a stack."""
import json

from . import storage
from .llm import LLMClient, ask_json, clean_mermaid
from .schemas import (
    WEIGHT_KEYS,
    FrameworkReport,
    ProjectInput,
    Recommendation,
    RecommendationBody,
    StackScore,
    required_keys,
)

NAMES = {"rag": "RAG", "n8n": "n8n", "crewai": "CrewAI", "autogen": "Autogen"}
ORDER = ["n8n", "rag", "crewai", "autogen"]


def stack_name(stack: list[str]) -> str:
    return " + ".join(NAMES[k] for k in sorted(stack, key=ORDER.index))


SYSTEM = (
    "You are the recommender (Agent 6) of a framework-selection workflow for SMALL or MICRO customer-support projects. "
    "You receive the project brief, the user's limits and weights, and one analysis report per framework the user chose. "
    "You decide which single framework or pair of frameworks to recommend, using only the information you are given. "
    "Write for a non-technical owner: plain language, honest about weaknesses and uncertainty."
)

FORMAT = """Reply with ONLY one JSON object, no other text:
{
  "ranking": [
    {"stack": ["n8n", "rag"],
     "covered": ["<requirement key>"], "missing": ["<requirement key>"],
     "estimated_daily_cost_usd": 0.0, "estimated_latency_ms": 0, "setup_hours": 0,
     "skill_required": "no-code | low-code | code", "expected_recall": null,
     "criteria": {"fit_to_requirements": 0, "cost": 0, "setup_effort": 0, "latency": 0, "long_term_benefit": 0},
     "reasoning": "one sentence"}
  ],
  "rationale": "3-5 sentences: why the top stack wins and why the main alternatives do not",
  "simple_design": "flowchart LR\\n    A[\\"...\\"] --> B[\\"...\\"]",
  "how_it_works": ["1) ...", "2) ..."],
  "phased_plan": ["Phase 1 (~2 h): ...", "..."],
  "upgrade_triggers": ["..."],
  "risks": ["..."],
  "recall_checklist": ["..."],
  "warnings": ["..."]
}

How to build "ranking":
- Candidates: EVERY framework in the reports appears alone as its own entry, plus every sensible pair (for example
  n8n + RAG). Never pair CrewAI with Autogen. Skip pairs that add nothing over their better member. At most 8 entries.
  A ranking with fewer entries than there are frameworks is wrong.
- For a pair, combine the reports realistically: fixed daily costs add up but the per-message LLM cost is shared;
  latency is the slowest member plus a small hand-over overhead; setup hours add up; "covered" is the union.
  Say how you combined them in "reasoning".
- Do NOT decide pass or fail yourself. The user's limits are applied afterwards to the numbers you give, so make the
  numbers accurate and consistent: "missing" lists the requirement keys the stack does not cover (empty if none),
  and cost, latency, setup hours and expected recall are your best combined estimates for the stack.
- "criteria" scores are 0-10 (10 best): fit_to_requirements = share of requirements covered; cost = cheaper within the
  budget is better (0 if over budget); setup_effort = less effort and a better match to user_skill is better; latency =
  faster is better (poor if over the limit); long_term_benefit = maintainability and extensibility. Be consistent
  across candidates. Do NOT compute a weighted total; the weights are applied afterwards.
- List the best candidate first.

The other fields describe the TOP stack:
- simple_design: one Mermaid flowchart ("flowchart LR", 5-8 nodes, every label in double quotes, plain --> arrows) of the
  combined stack built from THIS project's channels, documents, volume and approval step.
- how_it_works: 4-7 strings walking one real customer request through that design, naming concrete tools.
- phased_plan: concrete phases with hour estimates that add up to the stack's setup hours, ending with a go-live step
  where a person reviews the first ~50 replies.
- upgrade_triggers: 3-4 signs that it is time to move up or change (for example the message volume at which the daily
  budget breaks, which you can calculate from the reports).
- risks: the main practical risks (for example WhatsApp Business approval, Gmail OAuth).
- recall_checklist: if the project needs documents, 5-6 steps to test retrieval recall on the real documents against the
  minimum recall (20-30 realistic questions with known answers, count correct retrievals, what to tune if too low).
  Otherwise an empty list.
- warnings: anything the owner should be careful about with the top stack (for example a figure that is only a rough
  guess); otherwise an empty list."""


class RecommenderAgent:
    agent_id = "agent_6_recommender"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def recommend(
        self, project: ProjectInput, reports: list[FrameworkReport], assumptions: list[str]
    ) -> Recommendation:
        storage.log_event(self.agent_id, "running", f"scoring {len(reports)} reports")
        try:
            body = await ask_json(
                self.llm, RecommendationBody, self.agent_id, SYSTEM, self._prompt(project, reports), attempts=3,
                check=lambda b: self._structure_problems(b, reports),
            )
            rec = self._finish(body, project, reports, assumptions)
        except Exception as e:
            storage.log_event(self.agent_id, "error", str(e))
            raise
        storage.log_event(self.agent_id, "completed", f"recommended {stack_name(rec.recommended_stack)}")
        return rec

    # ------------------------------------------------------------------ prompt
    def _prompt(self, project: ProjectInput, reports: list[FrameworkReport]) -> str:
        r, c = project.requirements, project.constraints
        brief = {
            "project_description": project.project_description,
            "requirement_keys": required_keys(project),
            "messages_per_day": r.messages_per_day,
            "limits": {
                "budget_per_day_usd": c.budget_per_day_usd,
                "max_latency_ms": c.max_latency_ms,
                "min_recall": c.min_recall,
                "user_skill": c.user_skill,
                "time_available_hours": c.time_available_hours,
            },
            "weights": project.weights.model_dump(),
        }
        compact = [
            {
                "framework_key": rep.agent_id.split("_", 2)[2],
                "framework": rep.framework,
                "summary": rep.summary,
                "covered": rep.fit_to_requirements.covered,
                "missing": rep.fit_to_requirements.missing,
                "estimated_daily_cost_usd": rep.estimated_daily_cost_usd,
                "estimated_latency_ms": rep.estimated_latency_ms,
                "setup_hours": rep.setup_effort.hours,
                "skill_required": rep.setup_effort.skill_required,
                "long_term_score": rep.long_term_score,
                "expected_recall": rep.expected_recall,
                "risks": rep.risks,
                "works_well_with": rep.works_well_with,
                "assumptions": rep.assumptions,
            }
            for rep in reports
        ]
        return (
            f"PROJECT AND LIMITS (JSON):\n{json.dumps(brief, indent=2)}\n\n"
            f"FRAMEWORK REPORTS (JSON):\n{json.dumps(compact, indent=2)}\n\n{FORMAT}"
        )

    # ------------------------------------------------------- consistency only
    def _structure_problems(self, body: RecommendationBody, reports: list[FrameworkReport]) -> list[str]:
        """Complaints sent back to the model: every framework must appear alone, and only analyzed ones may be used."""
        known = {rep.agent_id.split("_", 2)[2] for rep in reports}
        alone = {s.stack[0] for s in body.ranking if len(s.stack) == 1}
        problems = []
        if known - alone:
            problems.append(f"the ranking must include each framework alone; missing: {sorted(known - alone)}")
        for s in body.ranking:
            if not set(s.stack) <= known:
                problems.append(f"{stack_name(s.stack)} uses a framework that was not analyzed; only use {sorted(known)}")
        return problems

    @staticmethod
    def _limit_flags(s: StackScore, project: ProjectInput) -> list[str]:
        """Which of the user's limits a stack breaks, judged from the AI's own numbers (plain comparisons)."""
        c, flags = project.constraints, []
        if s.missing:
            flags.append(f"does not cover: {', '.join(s.missing)}")
        if s.estimated_daily_cost_usd > c.budget_per_day_usd:
            flags.append(f"cost ${s.estimated_daily_cost_usd}/day exceeds the ${c.budget_per_day_usd}/day budget")
        if s.estimated_latency_ms > c.max_latency_ms:
            flags.append(f"latency ~{s.estimated_latency_ms} ms exceeds {c.max_latency_ms} ms")
        if project.requirements.document_upload and s.expected_recall is not None and s.expected_recall < c.min_recall:
            flags.append(f"expected recall {s.expected_recall} is below {c.min_recall}")
        if s.setup_hours > c.time_available_hours:
            flags.append(f"setup ~{s.setup_hours:g} h exceeds the {c.time_available_hours:g} h available")
        return flags

    def _finish(
        self, body: RecommendationBody, project: ProjectInput, reports: list[FrameworkReport], assumptions: list[str]
    ) -> Recommendation:
        """Keep the AI's content. Code only makes the numbers add up: it flags the limits the AI's numbers break, computes
        the weighted totals from the AI's criterion scores and the user's weights, and orders passing stacks first so the
        table matches the headline."""
        weights = project.weights.model_dump()
        known = {rep.agent_id.split("_", 2)[2] for rep in reports}
        ranking: list[StackScore] = []
        for s in body.ranking:
            if not set(s.stack) <= known:
                continue  # a stack that uses a framework the user did not ask about
            s.weighted_score = round(sum(s.criteria[k] * weights[k] for k in WEIGHT_KEYS), 2)
            s.flags = self._limit_flags(s, project)
            s.passes = not s.flags
            ranking.append(s)
        if not ranking:
            raise ValueError("the recommender returned no usable stack")
        ranking.sort(key=lambda s: (not s.passes, -s.weighted_score, len(s.stack)))
        for i, s in enumerate(ranking, 1):
            s.rank = i
        best = ranking[0]

        warnings = list(body.warnings)
        if not best.passes and not warnings:
            warnings.append("No option meets every hard limit; this is the best fit, not a pass. Issues: " + "; ".join(best.flags))
        merged = list(assumptions)
        for rep in reports:
            if rep.agent_id.split("_", 2)[2] in best.stack:
                merged += [a for a in rep.assumptions if a not in merged]

        return Recommendation(
            recommended_stack=best.stack,
            headline=f"Recommended: {stack_name(best.stack)}",
            rationale=body.rationale,
            ranking=ranking,
            comparison_table_md=self._table(ranking),
            simple_design=clean_mermaid(body.simple_design),
            how_it_works=body.how_it_works,
            phased_plan=body.phased_plan,
            estimated_daily_cost_usd=best.estimated_daily_cost_usd,
            estimated_monthly_cost_usd=round(best.estimated_daily_cost_usd * 30, 2),
            upgrade_triggers=body.upgrade_triggers,
            risks=body.risks,
            recall_checklist=body.recall_checklist,
            assumptions=merged,
            warnings=warnings,
        )

    def _table(self, ranking: list[StackScore]) -> str:
        rows = [
            "| Rank | Stack | Score | Fit | Cost | Latency | Effort | Hard limits |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in ranking[:8]:
            c = s.criteria
            rows.append(
                f"| {s.rank} | {stack_name(s.stack)} | {s.weighted_score} | {c['fit_to_requirements']} | "
                f"{c['cost']} (${s.estimated_daily_cost_usd}/day) | {c['latency']} (~{s.estimated_latency_ms} ms) | "
                f"{c['setup_effort']} (~{s.setup_hours:g} h) | {'pass' if s.passes else 'FAIL: ' + '; '.join(s.flags)} |"
            )
        return "\n".join(rows)
