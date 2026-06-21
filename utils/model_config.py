"""
Centralized model configuration.

  every agent reads its model from here. Swapping one agent to a bigger model
  Ollama listens on OLLAMA_HOST (default http://localhost:11434).

"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Base Ollama endpoint.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Global safety valve for your 8GB laptop: caps how many LLM calls run
# concurrently during the parallel per-paper fan-out. 
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "2"))

#Model Selection
SUMMARIZER_MODEL = "phi4-mini:latest"    #"llama3.1:8b"
QUALITY_ASSESSOR_MODEL = "phi4-mini:latest" #"llama3.1:8b"  # upgrade: "qwen2.5:14b"
CONTRADICTION_DETECTOR_MODEL = "phi4-mini:latest"  #"llama3.1:8b"  # upgrade: "qwen2.5:14b" or "phi4:14b"
SYNTHESIZER_MODEL = "phi4-mini:latest"  #"llama3.1:8b"  # synthesis needs a bigger context window, so we use the same model but with different config below



@dataclass
class AgentModelConfig:
    model: str
    temperature: float
    num_ctx: int = 4096  # context window; bump only if your hardware allows


# Per-agent model assignment.
# Right now everything points at llama3.1:8b, you can selectively bump the reasoning-heavy agents
# (quality_assessor, contradiction_detector) to a bigger model just by
# editing the strings below -- e.g. "qwen2.5:14b" -- with zero changes
# to agents/*.py.
MODEL_CONFIG = {
    "summarizer": AgentModelConfig(
        model=SUMMARIZER_MODEL,
        temperature=0.3,
    ),
    "quality_assessor": AgentModelConfig(
        model=QUALITY_ASSESSOR_MODEL,
        temperature=0.2,
    ),
    "contradiction_detector": AgentModelConfig(
        model=CONTRADICTION_DETECTOR_MODEL,
        temperature=0.2,
    ),
    "synthesizer": AgentModelConfig(
        model=SYNTHESIZER_MODEL,
        temperature=0.5,
        num_ctx=8192,  # synthesis needs a bigger context window
    ),
}


def get_model_config(agent_name: str) -> AgentModelConfig:
    """Look up the model config for a given agent. Raises a clear error
    instead of silently defaulting, so a typo'd agent name fails loudly
    during development rather than quietly using the wrong model."""
    if agent_name not in MODEL_CONFIG:
        raise KeyError(
            f"No model config for agent '{agent_name}'. "
            f"Available: {list(MODEL_CONFIG.keys())}"
        )
    return MODEL_CONFIG[agent_name]