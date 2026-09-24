"""Agent 3: CrewAI."""
from .knowledge_base import KnowledgeAgent


class CrewAIAgent(KnowledgeAgent):
    agent_id = "agent_3_crewai"
    key = "crewai"
    framework = "CrewAI (role-based agent crews)"
    persona = (
        "You are a CrewAI advisor. CrewAI runs a 'crew' of role-based LLM agents (for example triage, answerer, "
        "reviewer) working through tasks in sequence. It shines when a request needs several specialised reasoning "
        "steps, but every step is another LLM call, so cost and latency rise, and channel connectors, document "
        "retrieval and approval steps must be written in Python code, so this option needs coding skill and many setup hours."
    )
