"""Agent 5: Autogen."""
from .knowledge_base import KnowledgeAgent


class AutogenAgent(KnowledgeAgent):
    agent_id = "agent_5_autogen"
    key = "autogen"
    framework = "Autogen (conversational multi-agent framework)"
    persona = (
        "You are an Autogen guide (autogen-agentchat 0.4+). Autogen orchestrates LLM agents that converse, use tools "
        "and involve a human. It is very flexible, but for a small support bot the flexibility mostly means more code, "
        "more LLM calls per message and more latency than a workflow tool; connectors, retrieval and approval steps "
        "must be written in Python, so this option needs coding skill and many setup hours."
    )
