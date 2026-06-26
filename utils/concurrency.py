"""
Concurrency limiter for parallel LLM calls.

"""

import threading
from utils.model_config import MAX_CONCURRENT_LLM_CALLS

# A threading.Semaphore (not asyncio) since LangGraph's default
# execution model for sync node functions uses a thread pool, not an
# event loop. If the graph is later run in async mode, this would
# need to become an asyncio.Semaphore instead -- worth revisiting if
# we add async nodes later.
llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)