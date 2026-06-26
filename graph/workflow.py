"""
LangGraph workflow: sequential version.

This wires all six agents together in a straight pipeline, using the
typed ResearchState schema (graph/state.py) instead of a raw dict.

Deliberately built and tested before adding parallel fan-out (which
will replace the per-paper sections later) -- sequential is easier to
debug, and on an 8GB laptop with MAX_CONCURRENT_LLM_CALLS=2, the
practical benefit of parallelism is limited anyway. Get the graph
wiring and conditional routing correct first.

Fixes applied vs. the original plan:
- Uses StateGraph(ResearchState) instead of StateGraph(dict), so the
  typed schema we defined is actually enforced.
- Conditional routing: if ingestion returns zero papers, or parsing
  returns zero parseable papers, the graph routes straight to an
  early-exit node instead of continuing through summarize -> assess
  -> contradictions -> synthesize on an empty list, which would
  previously produce a confusing empty/error report several steps
  later instead of failing fast with a clear message.
- The contradiction detector is correctly skipped (not just
  internally no-op'd) when fewer than 2 papers are available --
  the agent already guards this internally (Step 5), but routing
  around it here also skips the embedding model load entirely.
"""

from langgraph.graph import StateGraph, END
from loguru import logger

from graph.state import ResearchState
from agents.ingestion import ingestion_node
from agents.parser import parser_node
from agents.summarizer import summarization_node
from agents.quality_assessor import quality_assessment_node
from agents.contradiction_detector import contradiction_detector_node
from agents.synthesizer import synthesis_node


def early_exit_node(state: ResearchState) -> dict:
    """Reached when there's nothing left to process. Produces a
    clear, honest report explaining why, instead of letting an empty
    paper list silently flow through five more agent calls.

    Returns a partial state update -- see agents/ingestion.py's
    ingestion_node docstring for why mutating and returning the whole
    incoming state breaks LangGraph's reducer fields.
    """
    errors = state.get("errors", [])
    reason = "; ".join(errors) if errors else "No papers were available to process."
    logger.error(f"Early exit: {reason}")

    report = (
        f"# Literature Review: {state.get('query', 'unknown query')}\n\n"
        f"**Could not generate a report.**\n\n"
        f"Reason: {reason}\n\n"
        f"Try a different search query, increase the number of papers requested, "
        f"or check your network connection to arXiv."
    )
    return {"final_report": report}


def route_after_ingestion(state: ResearchState) -> str:
    """If ingestion found nothing, skip straight to early exit rather
    than running parse/summarize/etc. on an empty list."""
    if not state.get("papers"):
        return "early_exit"
    return "parse"


def route_after_parsing(state: ResearchState) -> str:
    """If nothing was parseable (e.g. all PDFs were scanned images),
    skip straight to early exit."""
    if not state.get("parsed_papers"):
        return "early_exit"
    return "summarize"


def create_workflow():
    workflow = StateGraph(ResearchState)

    workflow.add_node("ingest", ingestion_node)
    workflow.add_node("parse", parser_node)
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("assess_quality", quality_assessment_node)
    workflow.add_node("detect_contradictions", contradiction_detector_node)
    workflow.add_node("synthesize", synthesis_node)
    workflow.add_node("early_exit", early_exit_node)

    workflow.set_entry_point("ingest")

    workflow.add_conditional_edges(
        "ingest",
        route_after_ingestion,
        {"early_exit": "early_exit", "parse": "parse"},
    )
    workflow.add_conditional_edges(
        "parse",
        route_after_parsing,
        {"early_exit": "early_exit", "summarize": "summarize"},
    )

    workflow.add_edge("summarize", "assess_quality")
    workflow.add_edge("assess_quality", "detect_contradictions")
    workflow.add_edge("detect_contradictions", "synthesize")
    workflow.add_edge("synthesize", END)
    workflow.add_edge("early_exit", END)

    return workflow.compile()


def run_pipeline(query: str, num_papers: int = 5) -> ResearchState:
    """Convenience entry point for running the full pipeline."""
    app = create_workflow()

    initial_state: ResearchState = {
        "query": query,
        "num_papers": num_papers,
        "papers": [],
        "parsed_papers": [],
        "summaries": {},
        "quality_scores": {},
        "contradictions": [],
        "final_report": "",
        "errors": [],
    }

    logger.info(f"Starting pipeline: query='{query}', num_papers={num_papers}")
    result = app.invoke(initial_state)
    logger.info("Pipeline complete.")

    return result


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "BERT language model"
    num_papers = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    result = run_pipeline(query, num_papers)

    print("\n" + "=" * 60)
    print("ERRORS LOGGED DURING RUN")
    print("=" * 60)
    for e in result.get("errors", []):
        print(f"  - {e}")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])