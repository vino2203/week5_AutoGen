import json

import pytest

from agents.autogen_agent import AutogenAgent
from agents.crewai_agent import CrewAIAgent
from agents.n8n_agent import N8nAgent
from agents.rag_agent import RagAgent
from agents.schemas import required_keys
from tests.fakes import FakeLLM

ALL = [RagAgent, N8nAgent, CrewAIAgent, AutogenAgent]


@pytest.mark.parametrize("cls", ALL)
async def test_each_agent_asks_the_model_and_returns_a_validated_report(cls, project, fake):
    report = await cls().analyze(project, fake)
    agent = cls()
    assert fake.calls == [agent.agent_id]
    assert report.agent_id == agent.agent_id and report.framework == agent.framework
    assert report.simple_design.startswith("flowchart")  # fences stripped
    assert len(report.how_it_works) >= 4


async def test_prompt_carries_the_projects_own_details(project, fake):
    await N8nAgent().analyze(project, fake)
    prompt = fake.prompts["agent_2_n8n"]
    for detail in ("whatsapp", "gmail", "100", project.project_description[:30]):
        assert detail in prompt
    assert json.dumps(required_keys(project)) in prompt


async def test_keys_the_model_gets_wrong_are_made_consistent(project, fake):
    report = await RagAgent().analyze(project, fake)
    keys = required_keys(project)
    assert "invented_key" not in report.fit_to_requirements.covered
    assert report.fit_to_requirements.covered == ["document_upload"]
    assert report.fit_to_requirements.missing == [k for k in keys if k != "document_upload"]  # unclaimed = not covered
    assert report.works_well_with == ["n8n"] or "rag" not in report.works_well_with  # no made-up names, no self
    assert "made_up" not in report.works_well_with


async def test_invalid_json_twice_raises(project):
    bad = FakeLLM(garbage_for={"agent_1_rag"})
    with pytest.raises(ValueError):
        await RagAgent().analyze(project, bad)
    assert bad.calls == ["agent_1_rag", "agent_1_rag"]  # one retry
