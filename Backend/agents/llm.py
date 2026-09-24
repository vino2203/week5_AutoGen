"""LLM layer on top of Autogen. Every agent needs it: there is no offline fallback."""
import json
import os
import re
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import load_config

T = TypeVar("T", bound=BaseModel)

KEY_HELP = "OPENAI_API_KEY is missing. Put it in Backend/.env (copy .env.example) and restart the backend."


class LLMNotConfigured(RuntimeError):
    pass


MAX_TOOL_ITERATIONS = 8  # model turns the manager may take: brief, analyses, recommendation, wrap-up, plus corrections


class LLMClient(Protocol):
    async def complete(self, agent_name: str, system_prompt: str, user_prompt: str) -> str: ...

    async def run_with_tools(self, agent_name: str, system_prompt: str, task: str, tools: list[Callable]) -> str: ...


class AutogenLLM:
    """One Autogen AssistantAgent call per completion (fresh agent, no shared state)."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key, self.model = api_key, model
        self.temperature, self.max_tokens = temperature, max_tokens

    def _client(self):
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        kwargs = dict(model=self.model, api_key=self.api_key, temperature=self.temperature, max_tokens=self.max_tokens)
        try:
            return OpenAIChatCompletionClient(**kwargs)
        except ValueError:  # a model name Autogen does not know yet: describe its capabilities ourselves
            info = {"vision": False, "function_calling": True, "json_output": True, "family": "unknown", "structured_output": False}
            return OpenAIChatCompletionClient(model_info=info, **kwargs)

    async def complete(self, agent_name: str, system_prompt: str, user_prompt: str) -> str:
        from autogen_agentchat.agents import AssistantAgent

        client = self._client()
        try:
            agent = AssistantAgent(agent_name, model_client=client, system_message=system_prompt)
            result = await agent.run(task=user_prompt)
            return str(result.messages[-1].content)
        finally:
            await client.close()


    async def run_with_tools(self, agent_name: str, system_prompt: str, task: str, tools: list[Callable]) -> str:
        """An Autogen AssistantAgent that decides for itself which tools to call, and in what order, until it is done.
        Several tool calls in one model turn run concurrently."""
        from autogen_agentchat.agents import AssistantAgent

        client = self._client()
        try:
            agent = AssistantAgent(
                agent_name, model_client=client, tools=tools, system_message=system_prompt,
                max_tool_iterations=MAX_TOOL_ITERATIONS, reflect_on_tool_use=False,
            )
            result = await agent.run(task=task)
            return str(result.messages[-1].content)
        finally:
            await client.close()


def api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return "" if key in ("", "your-key-here") else key


def get_llm() -> LLMClient:
    key = api_key()
    if not key:
        raise LLMNotConfigured(KEY_HELP)
    cfg = load_config()
    return AutogenLLM(key, os.getenv("LLM_MODEL") or cfg["model"], cfg["temperature"], cfg["max_tokens"])


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a reply. Fences and chatter around it are skipped by the brace scan;
    fences inside string values (a ```mermaid diagram, say) are left alone."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object in reply")


def clean_mermaid(diagram: str) -> str:
    """Drop a code fence (and its 'mermaid' tag) around a diagram."""
    return re.sub(r"```(?:mermaid)?", "", diagram).strip()


async def ask_json(
    llm: LLMClient, model: type[T], agent_name: str, system: str, user: str, attempts: int = 2,
    check: Callable[[T], list[str]] | None = None,
) -> T:
    """Ask for JSON matching `model`. Invalid JSON, or contradictions found by `check` (a list of complaints), are sent
    back to the model with a request to correct them, up to `attempts` tries."""
    prompt, last_err = user, None
    for _ in range(attempts):
        reply = await llm.complete(agent_name, system, prompt)
        try:
            out = model.model_validate(extract_json(reply))
            problems = check(out) if check else []
            if not problems:
                return out
            last_err = "; ".join(problems)
            prompt = f"{user}\n\nYour previous reply contradicts itself or the rules:\n- " + "\n- ".join(problems) + "\nReply with ONLY the corrected JSON object."
        except (ValueError, ValidationError) as e:
            last_err = e
            prompt = f"{user}\n\nYour previous reply was invalid ({e}). Reply with ONLY the corrected JSON object."
    raise ValueError(f"the model did not return a valid, consistent reply: {last_err}")
