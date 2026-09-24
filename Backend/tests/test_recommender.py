import json

import pytest

from agents.human_loop_agent import render_markdown
from agents.recommender import RecommenderAgent, stack_name
from agents.schemas import RecommendationBody, StackScore, Weights
from agents.autogen_agent import AutogenAgent
from agents.n8n_agent import N8nAgent
from agents.rag_agent import RagAgent


def score(stack, fit=5, cost=0.2, latency=800, hours=6, **kw):
    c = {"fit_to_requirements": fit, "cost": 5, "setup_effort": 5, "latency": 5, "long_term_benefit": 5}
    return StackScore(stack=stack, criteria=c, estimated_daily_cost_usd=cost, estimated_latency_ms=latency,
                      setup_hours=hours, **kw)


def body(*ranking, warnings=()):
    return RecommendationBody(ranking=list(ranking), rationale="r", simple_design="```mermaid\nflowchart LR\n A-->B\n```",
                              how_it_works=["1) a"], phased_plan=["p"], warnings=list(warnings))


async def reports(project, fake):
    return [await c().analyze(project, fake) for c in (RagAgent, N8nAgent, AutogenAgent)]


async def test_recommend_end_to_end(project, fake):
    rec = await RecommenderAgent(fake).recommend(project, await reports(project, fake), ["assumed x"])
    assert sorted(rec.recommended_stack) == ["n8n", "rag"] and rec.ranking[0].passes
    assert rec.headline == "Recommended: n8n + RAG"
    assert rec.estimated_monthly_cost_usd == pytest.approx(rec.estimated_daily_cost_usd * 30, abs=0.01)
    assert rec.assumptions[0] == "assumed x" and "rag assumption" in rec.assumptions
    assert rec.simple_design.startswith("flowchart")  # fences stripped
    assert "| 1 | n8n + RAG |" in rec.comparison_table_md
    assert "## Recommended: n8n + RAG" in render_markdown(rec)


async def test_prompt_contains_limits_weights_and_every_report(project, fake):
    await RecommenderAgent(fake).recommend(project, await reports(project, fake), [])
    p = fake.prompts["agent_6_recommender"]
    assert '"budget_per_day_usd": 2' in p and '"weights"' in p and "Never pair CrewAI with Autogen" in p
    assert all(k in p for k in ('"framework_key": "rag"', '"framework_key": "n8n"', '"framework_key": "autogen"'))


async def test_weighted_scores_are_recomputed_from_criteria_and_weights(project, fake):
    project.weights = Weights(fit_to_requirements=1, cost=0, setup_effort=0, latency=0, long_term_benefit=0).resolved({})
    rec = RecommenderAgent(fake)._finish(body(score(["rag"], fit=8)), project, await reports(project, fake), [])
    assert rec.ranking[0].weighted_score == 8.0  # only fit counts


async def test_passing_stacks_come_first_then_by_score(project, fake):
    rs = await reports(project, fake)
    rec = RecommenderAgent(fake)._finish(
        body(score(["rag"], fit=9, latency=3000), score(["n8n"], fit=5), score(["autogen"], fit=8)), project, rs, []
    )
    assert [s.stack for s in rec.ranking] == [["autogen"], ["n8n"], ["rag"]]
    assert [s.rank for s in rec.ranking] == [1, 2, 3] and rec.recommended_stack == ["autogen"]


async def test_stacks_using_unselected_frameworks_are_dropped(project, fake):
    rs = [await RagAgent().analyze(project, fake)]
    rec = RecommenderAgent(fake)._finish(body(score(["crewai"], fit=10), score(["rag"], fit=1)), project, rs, [])
    assert [s.stack for s in rec.ranking] == [["rag"]]
    with pytest.raises(ValueError, match="no usable stack"):
        RecommenderAgent(fake)._finish(body(score(["crewai"])), project, rs, [])


async def test_a_warning_is_added_when_nothing_passes(project, fake):
    rs = await reports(project, fake)
    rec = RecommenderAgent(fake)._finish(body(score(["rag"], latency=3000)), project, rs, [])
    assert "No option meets every hard limit" in rec.warnings[0] and "latency" in rec.warnings[0]
    given = RecommenderAgent(fake)._finish(body(score(["rag"], latency=3000), warnings=["mine"]), project, rs, [])
    assert given.warnings == ["mine"]  # the model's own warning is kept


def test_criteria_must_be_complete_and_in_range():
    with pytest.raises(ValueError, match="criteria must contain"):
        StackScore(stack=["rag"], criteria={"cost": 5}, estimated_daily_cost_usd=0, estimated_latency_ms=0, setup_hours=0)
    with pytest.raises(ValueError, match="between 0 and 10"):
        StackScore(stack=["rag"], criteria={"fit_to_requirements": 11, "cost": 5, "setup_effort": 5, "latency": 5, "long_term_benefit": 5},
                   estimated_daily_cost_usd=0, estimated_latency_ms=0, setup_hours=0)


def test_stack_name_orders_consistently():
    assert stack_name(["rag", "n8n"]) == "n8n + RAG"


class Scripted:
    """Replies to the recommender with prepared JSON, one per call."""

    def __init__(self, *ranking_lists):
        self.replies, self.prompts = [json.dumps(self._body(r)) for r in ranking_lists], []

    @staticmethod
    def _body(ranking):
        return {"ranking": ranking, "rationale": "r", "simple_design": "flowchart LR\n A-->B", "how_it_works": ["1) a"], "phased_plan": ["p"]}

    async def complete(self, agent_name, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return self.replies.pop(0)


def entry(stack, cost=0.2, latency=800, hours=6, flags=(), missing=(), recall=None):
    return {"stack": stack, "covered": [], "missing": list(missing), "estimated_daily_cost_usd": cost,
            "estimated_latency_ms": latency, "setup_hours": hours, "skill_required": "low-code", "expected_recall": recall,
            "criteria": {"fit_to_requirements": 5, "cost": 5, "setup_effort": 5, "latency": 5, "long_term_benefit": 5},
            "flags": list(flags)}


async def test_the_ranking_must_include_every_framework_alone(project, fake):
    rs = [await RagAgent().analyze(project, fake), await N8nAgent().analyze(project, fake)]
    llm = Scripted([entry(["rag"])], [entry(["rag"]), entry(["n8n"])])
    rec = await RecommenderAgent(llm).recommend(project, rs, [])
    assert len(rec.ranking) == 2 and "must include each framework alone" in llm.prompts[1]


async def test_limits_are_applied_to_the_ais_numbers_and_its_own_flags_are_ignored(project, fake):
    rs = [await RagAgent().analyze(project, fake)]
    broken = entry(["rag"], cost=5.0, latency=3000, hours=90, recall=0.5, missing=["document_upload"], flags=["looks fine"])
    llm = Scripted([broken])
    rec = await RecommenderAgent(llm).recommend(project, rs, [])
    flags = " | ".join(rec.ranking[0].flags)
    for word in ("does not cover: document_upload", "cost $5.0/day exceeds the $2.0/day budget", "latency ~3000 ms exceeds 1000 ms",
                 "expected recall 0.5 is below 0.9", "setup ~90 h exceeds the 40 h available"):
        assert word in flags
    assert "looks fine" not in flags and not rec.ranking[0].passes


async def test_a_stack_within_every_limit_passes_whatever_the_ai_claims(project, fake):
    rs = [await RagAgent().analyze(project, fake)]
    fine = entry(["rag"], cost=0.4, latency=900, hours=10, recall=0.95, flags=["cost far too high"])
    rec = await RecommenderAgent(Scripted([fine])).recommend(project, rs, [])
    assert rec.ranking[0].flags == [] and rec.ranking[0].passes


async def test_recall_only_counts_when_the_project_needs_documents(project, fake):
    project.requirements.document_upload = False
    rs = [await RagAgent().analyze(project, fake)]
    rec = await RecommenderAgent(Scripted([entry(["rag"], recall=0.1)])).recommend(project, rs, [])
    assert rec.ranking[0].passes


async def test_a_recommender_that_omits_frameworks_is_an_error_after_three_tries(project, fake):
    rs = [await RagAgent().analyze(project, fake), await N8nAgent().analyze(project, fake)]
    only_rag = [entry(["rag"])]
    with pytest.raises(ValueError, match="must include each framework alone"):
        await RecommenderAgent(Scripted(only_rag, only_rag, only_rag)).recommend(project, rs, [])
