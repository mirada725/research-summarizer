"""
Shared state schema for the LangGraph workflow.

Fixes applied vs. the original plan:
- Uses a proper TypedDict (ResearchState) passed to StateGraph,
  instead of `StateGraph(dict)`. The original plan defined
  ResearchState but then didn't actually use it when compiling the
  graph -- StateGraph(dict) loses all the type-safety benefit of
  having defined the schema in the first place.
- Uses Annotated + operator.add for fields that get populated by
  parallel fan-out branches (parsed_papers, summaries, etc.) so
  LangGraph knows to MERGE results from concurrent branches rather
  than overwrite -- required for the Send-based parallel fan-out in
  graph/workflow.py. Fields that are simple overwrites (final_report,
  query) are left as plain types.
"""

from typing import TypedDict, Annotated
import operator


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer for dict-valued state fields populated by parallel
    branches (e.g. summaries, quality_scores keyed by paper title).
    Each parallel branch returns a single-key dict; LangGraph calls
    this to merge them as branches complete, instead of the default
    behavior of the last writer overwriting everyone else's results."""
    return {**left, **right}


class ResearchState(TypedDict):
    # User inputs -- set once at the start, never mutated mid-run
    query: str
    num_papers: int

    # Ingestion output
    papers: list[dict]

    # Per-paper parallel fan-out outputs. Annotated with operator.add
    # (for the list) and merge_dicts (for the dicts keyed by title)
    # so concurrent branches accumulate instead of overwrite.
    parsed_papers: Annotated[list[dict], operator.add]
    summaries: Annotated[dict, merge_dicts]
    quality_scores: Annotated[dict, merge_dicts]

    # Cross-paper outputs -- single writer, no merge needed
    contradictions: list[dict]
    final_report: str

    # Metadata / observability
    errors: Annotated[list[str], operator.add]