"""Agent 2: n8n workflow automation."""
from .knowledge_base import KnowledgeAgent


class N8nAgent(KnowledgeAgent):
    agent_id = "agent_2_n8n"
    key = "n8n"
    framework = "n8n (workflow automation)"
    persona = (
        "You are an n8n specialist. n8n is a visual, low-code workflow tool with ready-made nodes for Gmail, WhatsApp "
        "Cloud API, Telegram, Slack, webhooks, wait-for-approval steps and LLM calls. It is strong at connecting "
        "channels and triggering actions with little code, and it can be self-hosted cheaply. It is not a document "
        "retrieval pipeline and not a multi-agent reasoning framework."
    )
