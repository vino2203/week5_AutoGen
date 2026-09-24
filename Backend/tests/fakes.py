"""A fake model for tests. Only the network is faked:

- the manager runs through the REAL Autogen tool loop (AutogenLLM.run_with_tools), fed by a scripted model that "decides"
  which tools to call (autogen's ReplayChatCompletionClient);
- the framework agents and the recommender get canned JSON from `complete`.

It checks the plumbing, not the quality of real AI output.
"""
import asyncio
import json

from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from agents.llm import AutogenLLM
from agents.schemas import Brief, Constraints

ALL = ["rag", "n8n", "crewai", "autogen"]
MODEL_INFO = {"vision": False, "function_calling": True, "json_output": True, "family": "unknown", "structured_output": False}

BRIEF_ARGS = {
    "options_to_compare": ALL, "channels": ["gmail", "sms"], "document_upload": True, "messages_per_day": 50,
    "human_approval_of_replies": False, "multi_step_reasoning": False, "budget_per_day_usd": 2, "max_latency_ms": 1000,
    "min_recall": 0.9, "user_skill": "low-code", "time_available_hours": 40,
    "assumptions": ["Read the channels from the description."],
}


def standard_brief() -> Brief:
    args = dict(BRIEF_ARGS)
    limits = {k: args.pop(k) for k in ("budget_per_day_usd", "max_latency_ms", "min_recall", "user_skill", "time_available_hours")}
    return Brief(constraints=Constraints(**limits), **args)


def calls(*tool_calls) -> CreateResult:
    """One model turn that calls tools: calls(("analyze_framework", {"option": "rag"}), ...)."""
    return CreateResult(
        finish_reason="function_calls", cached=False, usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
        content=[FunctionCall(id=str(i), name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(tool_calls)],
    )


def happy_script(options=ALL, brief=None):
    """What a well-behaved manager model does: brief, then all analyses in one step, then the recommendation."""
    return [
        calls(("set_project_brief", brief or BRIEF_ARGS)),
        calls(*[("analyze_framework", {"option": o}) for o in options]),
        calls(("recommend_stack", {})),
        "Recommended a stack.",
    ]


REPORTS = {
    "rag": dict(covers=["document_upload"], hours=10, skill="low-code", cost=0.0875, latency=900, lt=7, recall=0.9),
    "n8n": dict(covers=["channel", "human_approval"], hours=6, skill="low-code", cost=0.2, latency=800, lt=8, recall=None),
    "crewai": dict(covers=["channel", "document_upload", "human_approval", "multi_step_reasoning"], hours=24, skill="code", cost=0.32, latency=4500, lt=6, recall=0.9),
    "autogen": dict(covers=["channel", "document_upload", "human_approval", "multi_step_reasoning"], hours=28, skill="code", cost=0.28, latency=3000, lt=7, recall=0.9),
}
DESIGN = 'flowchart LR\n    U["Customer (Gmail)"] --> N["Workflow"]\n    N --> L["LLM writes the reply"]\n    L --> R["Send reply"]\n    R --> U'


class FakeLLM(AutogenLLM):
    def __init__(self, scripts=None, garbage_for=(), fail_for=()):
        super().__init__("sk-fake", "fake-model", 0.0, 100)
        self.scripts = scripts  # a list of manager scripts, one per manager run; None = the well-behaved script every time
        self.calls: list[str] = []  # framework agents and the recommender (not the manager)
        self.prompts: dict[str, str] = {}
        self.manager_tasks: list[str] = []
        self.garbage_for, self.fail_for = set(garbage_for), set(fail_for)
        self.active = self.max_active = 0  # how many complete() calls overlap: shows the agents really run in parallel

    # ------------------------------------------------------------------ manager
    def _client(self):
        script = self.scripts.pop(0) if self.scripts else happy_script()
        return ReplayChatCompletionClient(script, model_info=MODEL_INFO)

    async def run_with_tools(self, agent_name, system_prompt, task, tools):
        self.manager_tasks.append(task)
        return await super().run_with_tools(agent_name, system_prompt, task, tools)

    # ---------------------------------------------------- framework agents, agent 6
    async def complete(self, agent_name, system_prompt, user_prompt):
        self.calls.append(agent_name)
        self.prompts[agent_name] = user_prompt
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            if agent_name in self.fail_for:
                raise RuntimeError("model unavailable")
            if agent_name in self.garbage_for:
                return "sorry, no JSON here"
            if agent_name == "agent_6_recommender":
                return "```json\n" + json.dumps(self._recommend(user_prompt)) + "\n```"
            return json.dumps(self._report(agent_name.split("_", 2)[2], user_prompt))
        finally:
            self.active -= 1

    def _report(self, key, prompt):
        keys = json.loads(prompt.split("REQUIREMENT KEYS")[1].split(":", 1)[1].split("\n")[0].strip())
        spec = REPORTS[key]
        covered = [k for k in keys if k.split(":")[0] in spec["covers"]]
        # deliberately sloppy like a real model: an invented key, and requirements left unmentioned
        return {
            "summary": f"{key} summary",
            "fit_to_requirements": {"covered": covered + ["invented_key"], "missing": []},
            "setup_effort": {"hours": spec["hours"], "skill_required": spec["skill"], "steps": [{"step": "Set it up", "hours": spec["hours"]}]},
            "estimated_daily_cost_usd": spec["cost"],
            "estimated_latency_ms": spec["latency"],
            "long_term_score": spec["lt"],
            "expected_recall": spec["recall"],
            "risks": [f"{key} risk"],
            "works_well_with": ["rag", "made_up", key],
            "simple_design": "```mermaid\n" + DESIGN + "\n```",
            "how_it_works": ["1) a", "2) b", "3) c", "4) d"],
            "project_specific_notes": ["note"],
            "assumptions": [f"{key} assumption"],
        }

    def _recommend(self, prompt):
        limits = json.loads(prompt.split("PROJECT AND LIMITS (JSON):\n")[1].split("\n\nFRAMEWORK REPORTS")[0])["limits"]
        reports = json.loads(prompt.split("FRAMEWORK REPORTS (JSON):\n")[1].split("\n\nReply with ONLY")[0])
        by = {r["framework_key"]: r for r in reports}
        stacks = [[k] for k in by] + ([["n8n", "rag"]] if "n8n" in by and "rag" in by else [])
        ranking = []
        for st in stacks:
            rs = [by[k] for k in st]
            cost = sum(r["estimated_daily_cost_usd"] for r in rs)
            latency = max(r["estimated_latency_ms"] for r in rs) + 50 * (len(rs) - 1)
            covered = sorted({c for r in rs for c in r["covered"]})
            missing = sorted({m for r in rs for m in r["missing"]} - set(covered))
            flags = []
            if missing:
                flags.append("does not cover: " + ", ".join(missing))
            if cost > limits["budget_per_day_usd"]:
                flags.append(f"cost ${round(cost, 4)}/day exceeds the budget")
            if latency > limits["max_latency_ms"]:
                flags.append(f"latency ~{latency} ms exceeds {limits['max_latency_ms']} ms")
            ranking.append({
                "stack": st, "covered": covered, "missing": missing,
                "estimated_daily_cost_usd": round(cost, 4), "estimated_latency_ms": latency,
                "setup_hours": sum(r["setup_hours"] for r in rs), "skill_required": "low-code", "expected_recall": None,
                "criteria": {"fit_to_requirements": 10 * len(covered) / max(1, len(covered) + len(missing)),
                             "cost": 9, "setup_effort": 6, "latency": 0 if latency > limits["max_latency_ms"] else 6, "long_term_benefit": 7},
                "flags": flags, "reasoning": "combined by the fake model",
            })
        ranking.sort(key=lambda s: (bool(s["flags"]), len(s["stack"])))  # deliberately not sorted by weighted score
        return {
            "ranking": ranking, "rationale": "Because the fake model says so.",
            "simple_design": DESIGN, "how_it_works": ["1) a", "2) b", "3) c", "4) d"],
            "phased_plan": ["Phase 1 (~2 h): start", "Phase 2: go live"], "upgrade_triggers": ["volume grows"],
            "risks": ["a risk"], "recall_checklist": ["write 20 questions"], "warnings": [],
        }
