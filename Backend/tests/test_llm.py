import pytest

from agents.llm import AutogenLLM, LLMNotConfigured, ask_json, clean_mermaid, extract_json, get_llm
from agents.schemas import Constraints


def test_extract_json_handles_fences_chatter_and_braces_in_strings():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nDone') == {"a": 1}
    assert extract_json('{"a": "x } y", "b": {"c": 2}} trailing') == {"a": "x } y", "b": {"c": 2}}
    fenced_inside = '{"d": "```mermaid\\nflowchart LR\\n A-->B\\n```"}'
    assert extract_json(fenced_inside)["d"].startswith("```mermaid")  # not damaged by the extraction
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_clean_mermaid_strips_fences():
    assert clean_mermaid("```mermaid\nflowchart LR\n A-->B\n```") == "flowchart LR\n A-->B"


def test_missing_or_placeholder_key_is_a_clear_error(monkeypatch):
    with pytest.raises(LLMNotConfigured, match="Backend/.env"):
        get_llm()
    monkeypatch.setenv("OPENAI_API_KEY", "your-key-here")
    with pytest.raises(LLMNotConfigured):
        get_llm()


def test_key_from_env_builds_the_autogen_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    llm = get_llm()
    assert isinstance(llm, AutogenLLM) and llm.model == "gpt-4o-mini"
    assert llm._client() is not None


def test_unknown_model_names_still_build_a_client():
    assert AutogenLLM("sk-test", "some-future-model", 0.2, 100)._client() is not None


class Scripted:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    async def complete(self, agent_name, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return self.replies.pop(0)


async def test_ask_json_retries_once_with_the_error_then_succeeds():
    llm = Scripted(["nope", '{"budget_per_day_usd": 3}'])
    out = await ask_json(llm, Constraints, "a", "sys", "user")
    assert out.budget_per_day_usd == 3
    assert "previous reply was invalid" in llm.prompts[1]


async def test_ask_json_gives_up_after_two_bad_replies():
    llm = Scripted(["nope", '{"budget_per_day_usd": -1}'])
    with pytest.raises(ValueError, match="valid, consistent reply"):
        await ask_json(llm, Constraints, "a", "sys", "user")
    assert len(llm.prompts) == 2


async def test_ask_json_sends_contradictions_back_to_the_model():
    llm = Scripted(['{"budget_per_day_usd": 1}', '{"budget_per_day_usd": 2}'])
    out = await ask_json(llm, Constraints, "a", "sys", "user",
                         check=lambda c: [] if c.budget_per_day_usd == 2 else ["the budget must be 2"])
    assert out.budget_per_day_usd == 2
    assert "contradicts itself" in llm.prompts[1] and "the budget must be 2" in llm.prompts[1]


async def test_ask_json_gives_up_if_it_stays_inconsistent():
    llm = Scripted(['{"budget_per_day_usd": 1}'] * 3)
    import pytest as _p
    with _p.raises(ValueError, match="the budget must be 2"):
        await ask_json(llm, Constraints, "a", "sys", "user", attempts=3, check=lambda c: ["the budget must be 2"])
    assert len(llm.prompts) == 3
