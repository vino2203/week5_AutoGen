"""Agent 1: Retrieval-Augmented Generation."""
from .knowledge_base import KnowledgeAgent


class RagAgent(KnowledgeAgent):
    agent_id = "agent_1_rag"
    key = "rag"
    framework = "RAG (Retrieval-Augmented Generation)"
    persona = (
        "You are an expert on Retrieval-Augmented Generation for customer support. RAG retrieves the most relevant "
        "chunks of the owner's documents and gives them to an LLM so answers stay grounded. It is a technique, not a "
        "channel connector: on its own it does not receive Gmail/WhatsApp messages or send replies. Quality depends on "
        "chunking, embeddings, top-k and keeping documents up to date. Used ALONE, RAG covers only answering from documents: "
        "channels (Gmail, WhatsApp, ...), sending replies and human approval are NOT covered, so they are missing."
    )
