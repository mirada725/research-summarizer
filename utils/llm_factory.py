"""
Factory for getting a configured LLM instance per agent.

Every agent should call get_llm("agent_name") instead of constructing
OllamaLLM directly. This is the single place that knows about
OLLAMA_BASE_URL, per-agent model/temperature, and context window
"""

from langchain_ollama import OllamaLLM
from utils.model_config import get_model_config, OLLAMA_BASE_URL


def get_llm(agent_name: str) -> OllamaLLM:
    cfg = get_model_config(agent_name)
    return OllamaLLM(
        model=cfg.model,
        base_url=OLLAMA_BASE_URL,
        temperature=cfg.temperature,
        num_ctx=cfg.num_ctx,
    )