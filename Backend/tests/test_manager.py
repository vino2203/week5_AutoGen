from agents.config import load_config
from agents.manager import Manager
from agents.schemas import Constraints, ProjectInput, Weights
from tests.fakes import BRIEF_ARGS, FakeLLM, calls, happy_script

KNOWLEDGE = {"agent_1_rag", "agent_2_n8n", "agent_3_crewai", "agent_5_autogen"}


def run_manager(fake):
    return Manager(fake, load_config())


async def test_the_ai_manager_runs_the_whole_workflow(manager, fake, sample_input):
    session = await manager.start(sample_input)
    assert session.status == "awaiting_approval" and not session.errors
    assert {r.agent_id for r in session.reports} == KNOWLEDGE
    assert sorted(session.recommendation.recommended_stack) == ["n8n", "rag"]
    assert fake.calls.count("agent_6_recommender") == 1
    assert fake.max_active >= 4  # the analyses the manager requested in one step ran in parallel
    assert len(fake.manager_tasks) == 1 and "USER INPUT" in fake.manager_tasks[0]


async def test_the_brief_comes_from_the_manager_model(manager, sample_input):
    session = await manager.start(ProjectInput(project_description="Order status by email and SMS, 50 a day"))
    r = session.project.requirements
    assert r.channels == ["gmail", "sms"] and r.document_upload is True and r.messages_per_day == 50
    assert "Read the channels from the description." in session.assumptions
    assert any("weights" in a for a in session.assumptions)
    assert session.project.constraints.max_latency_ms == 1000


async def test_the_manager_can_choose_a_subset(sample_input):
    brief = {**BRIEF_ARGS, "options_to_compare": ["rag", "n8n"]}
    fake = FakeLLM(scripts=[happy_script(["rag", "n8n"], brief)])
    session = await run_manager(fake).start(ProjectInput(project_description="compare rag and n8n please"))
    assert {r.agent_id for r in session.reports} == {"agent_1_rag", "agent_2_n8n"}
    assert set(fake.calls) == {"agent_1_rag", "agent_2_n8n", "agent_6_recommender"}


async def test_options_the_user_chose_cannot_be_widened_by_the_manager(sample_input):
    sample_input.options_to_compare = ["rag"]
    fake = FakeLLM()  # the model's brief says all four and it asks for all four
    session = await run_manager(fake).start(sample_input)
    assert session.project.options_to_compare == ["rag"]
    assert [r.agent_id for r in session.reports] == ["agent_1_rag"]
    assert set(fake.calls) == {"agent_1_rag", "agent_6_recommender"}  # the other requests were refused by the tool


async def test_what_the_user_set_explicitly_beats_the_managers_reading(manager):
    p = ProjectInput.model_validate({
        "project_description": "support bot",
        "requirements": {"channels": ["telegram"], "document_upload": False, "messages_per_day": 7},
        "constraints": {"budget_per_day_usd": 0.5},
        "weights": {"cost": 5},
    })
    session = await manager.start(p)
    r, c = session.project.requirements, session.project.constraints
    assert r.channels == ["telegram"] and r.document_upload is False and r.messages_per_day == 7
    assert c.budget_per_day_usd == 0.5 and c.max_latency_ms == 1000
    assert session.project.weights.cost > session.project.weights.latency


async def test_empty_description_asks_and_the_manager_is_not_run(manager, fake):
    session = await manager.start(ProjectInput())
    assert session.status == "needs_input" and len(session.questions) == 1
    assert fake.manager_tasks == [] and fake.calls == []


async def test_an_invalid_brief_is_sent_back_to_the_model_which_corrects_it(sample_input):
    bad_range = {**BRIEF_ARGS, "min_recall": 5}  # rejected by our validation
    bad_choice = {**BRIEF_ARGS, "user_skill": "expert"}  # rejected by the tool schema
    script = [calls(("set_project_brief", bad_range)), calls(("set_project_brief", bad_choice))] + happy_script()
    fake = FakeLLM(scripts=[script])
    session = await run_manager(fake).start(sample_input)
    assert session.status == "awaiting_approval" and len(session.reports) == 4


async def test_calling_the_same_analysis_twice_runs_it_once(sample_input):
    script = [
        calls(("set_project_brief", BRIEF_ARGS)),
        calls(("analyze_framework", {"option": "rag"}), ("analyze_framework", {"option": "rag"})),
        calls(*[("analyze_framework", {"option": o}) for o in ["n8n", "crewai", "autogen"]]),
        calls(("recommend_stack", {})),
        "ok",
    ]
    fake = FakeLLM(scripts=[script])
    await run_manager(fake).start(sample_input)
    assert fake.calls.count("agent_1_rag") == 1


async def test_a_manager_that_stops_early_is_reminded_once_and_finishes(sample_input):
    first = happy_script()[:2] + ["I think that is enough."]  # brief and analyses, but no recommendation
    second = [calls(("recommend_stack", {})), "Now done."]
    fake = FakeLLM(scripts=[first, second])
    session = await run_manager(fake).start(sample_input)
    assert session.status == "awaiting_approval" and session.recommendation is not None
    assert len(fake.manager_tasks) == 2
    assert "PROGRESS SO FAR" in fake.manager_tasks[1] and "recommendation: NOT done yet" in fake.manager_tasks[1]
    assert fake.calls.count("agent_2_n8n") == 1  # nothing was analyzed twice


async def test_a_manager_that_never_recommends_is_an_error_not_a_guess(sample_input):
    stops = happy_script()[:2] + ["done"]
    fake = FakeLLM(scripts=[stops, ["still nothing"]])
    session = await run_manager(fake).start(sample_input)
    assert session.status == "error" and session.recommendation is None
    assert "did not ask for a recommendation" in session.errors[0].error
    assert len(session.reports) == 4  # the analyses are kept


async def test_a_manager_that_never_saves_a_brief_is_an_error(sample_input):
    fake = FakeLLM(scripts=[["I cannot help with that."], ["Still cannot."]])
    session = await run_manager(fake).start(sample_input)
    assert session.status == "error"
    assert "did not save a project brief" in session.errors[0].error
    assert fake.calls == []


async def test_a_failing_framework_agent_is_retried_then_excluded(sample_input):
    fake = FakeLLM(fail_for={"agent_2_n8n"})
    session = await run_manager(fake).start(sample_input)
    assert fake.calls.count("agent_2_n8n") == 2  # one retry
    assert [e.agent_id for e in session.errors] == ["agent_2_n8n"]
    assert "agent_2_n8n" not in {r.agent_id for r in session.reports}
    assert session.status == "awaiting_approval" and session.recommendation is not None


async def test_all_analyses_failing_is_an_error(sample_input):
    fake = FakeLLM(fail_for=KNOWLEDGE, scripts=[happy_script(), happy_script()])
    session = await run_manager(fake).start(sample_input)
    assert session.status == "error"
    assert {e.agent_id for e in session.errors} >= KNOWLEDGE
    assert "agent_6_recommender" not in fake.calls


async def test_a_failing_recommender_is_reported_by_name(sample_input):
    fake = FakeLLM(garbage_for={"agent_6_recommender"}, scripts=[happy_script(), [calls(("recommend_stack", {})), "x"]])
    session = await run_manager(fake).start(sample_input)
    assert session.status == "error" and session.errors[-1].agent_id == "agent_6_recommender"
    assert len(session.reports) == 4


async def test_a_model_or_network_failure_is_reported(sample_input):
    class Down(FakeLLM):
        async def run_with_tools(self, *a, **k):
            raise RuntimeError("401 invalid api key")

    session = await run_manager(Down()).start(sample_input)
    assert session.status == "error" and "invalid api key" in session.errors[0].error


async def test_revise_only_asks_the_recommender_again(manager, fake, sample_input):
    session = await manager.start(sample_input)
    before = len(fake.calls)
    session = await manager.revise(session, Constraints(budget_per_day_usd=0.05), Weights(cost=90))
    assert fake.calls[before:] == ["agent_6_recommender"] and len(fake.manager_tasks) == 1
    assert '"budget_per_day_usd": 0.05' in fake.prompts["agent_6_recommender"]
    assert session.revision == 1 and session.project.weights.cost > 0.6
    assert session.recommendation.ranking[0].flags  # the fake flags everything as over budget


async def test_session_is_persisted_and_decisions_are_stored(manager, sample_input):
    session = await manager.start(sample_input)
    assert manager.load(session.session_id).recommendation.headline == session.recommendation.headline
    assert manager.decide(session, approve=True).status == "approved"
    assert manager.load(session.session_id).status == "approved"
