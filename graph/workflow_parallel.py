"""
LangGraph workflow: parallel fan-out version.

Kept as a SEPARATE file from graph/workflow.py (the sequential
version) rather than replacing it -- this lets you compare behavior,
fall back to sequential for debugging, and avoids risking the proven
working baseline while this newer, more complex version gets exercised
against real data for the first time.

What's different from the sequential version:
- summarize and assess_quality each fan out to one Send-dispatched
  node PER PAPER (agents.summarizer.summarize_one_paper_node and
  agents.quality_assessor.assess_one_paper_node), instead of one node
  looping over all papers internally.
- Real concurrency is capped by utils.concurrency.llm_semaphore,
  sized from MAX_CONCURRENT_LLM_CALLS in utils/model_config.py (=2 by
  default, tuned for an 8GB laptop). LangGraph's own fan-out has no
  awareness of Ollama's capacity, so without this semaphore, a
  5-paper run would fire 5 concurrent LLM calls at once.
- The "merge point" after fan-out (LangGraph waits for all Send
  branches to complete before continuing) is implicit -- assess_quality
  only starts once every summarize_one_paper_node branch has returned,
  same as detect_contradictions only starts once every assess branch
  has returned. No explicit "join" node is needed; LangGraph's graph
  execution model handles this based on the edges defined below.

Everything else (conditional early-exit routing, agent logic) is
identical to graph/workflow.py.
"""

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from loguru import logger

from graph.state import ResearchState
from agents.ingestion import ingestion_node
from agents.parser import parser_node
from agents.summarizer import summarize_one_paper_node
from agents.quality_assessor import assess_one_paper_node
from agents.contradiction_detector import contradiction_detector_node
from agents.synthesizer import synthesis_node


def early_exit_node(state: ResearchState) -> dict:
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
    if not state.get("papers"):
        return "early_exit"
    return "parse"


def route_after_parsing_for_fanout(state: ResearchState):
    """Combined routing + fan-out function for the post-parse edge.

    LangGraph's add_conditional_edges expects the routing function's
    return value to be either: a plain string naming the next node
    (looked up via path_map), OR a list of Send objects that carry
    their own target node names directly -- the two cannot be mixed
    in one path_map, since path_map values must be node-name strings,
    not callables. So this single function handles both outcomes
    itself: returns the early_exit string for the empty case, or a
    list of Send objects to fan out to "summarize_one" otherwise.
    """
    if not state.get("parsed_papers"):
        return "early_exit"
    return [Send("summarize_one", {"paper": paper}) for paper in state["parsed_papers"]]


def fan_out_to_assessors(state: ResearchState) -> list[Send]:
    """Same pattern as fan_out_to_summarizers, for quality assessment.
    Runs after all summarize_one branches have completed (since this
    is reached via a normal edge from a node, not from inside the
    fan-out itself), reusing parsed_papers as the source list -- the
    assessor reads from sections/abstract, not from the summary, so
    it doesn't need to wait on summaries content, only on graph order.
    """
    return [
        Send("assess_one", {"paper": paper})
        for paper in state["parsed_papers"]
    ]


def create_workflow():
    workflow = StateGraph(ResearchState)

    workflow.add_node("ingest", ingestion_node)
    workflow.add_node("parse", parser_node)
    workflow.add_node("summarize_one", summarize_one_paper_node)
    workflow.add_node("assess_one", assess_one_paper_node)
    workflow.add_node("detect_contradictions", contradiction_detector_node)
    workflow.add_node("synthesize", synthesis_node)
    workflow.add_node("early_exit", early_exit_node)

    workflow.set_entry_point("ingest")

    workflow.add_conditional_edges(
        "ingest",
        route_after_ingestion,
        {"early_exit": "early_exit", "parse": "parse"},
    )

    # After parsing, conditionally either early-exit or fan out to
    # parallel summarization. add_conditional_edges' path map values
    # can themselves be the fan-out function for the "summarize" case,
    # letting LangGraph treat the Send list as the next step.
    workflow.add_conditional_edges(
        "parse",
        lambda state: "early_exit" if not state.get("parsed_papers") else "fan_out_summarize",
        {
            "early_exit": "early_exit",
            "fan_out_summarize": fan_out_to_summarizers,
        },
    )

    # All summarize_one branches converge here (LangGraph waits for
    # every Send dispatched from the parse->fan_out_summarize edge to
    # complete before this edge fires), then fan out again for quality
    # assessment.
    workflow.add_conditional_edges(
        "summarize_one",
        fan_out_to_assessors,
    )

    # All assess_one branches converge before contradictions/synthesis.
    workflow.add_edge("assess_one", "detect_contradictions")
    workflow.add_edge("detect_contradictions", "synthesize")
    workflow.add_edge("synthesize", END)
    workflow.add_edge("early_exit", END)

    return workflow.compile()


def run_pipeline(query: str, num_papers: int = 5) -> ResearchState:
    """Convenience entry point for running the full parallel pipeline."""
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

    logger.info(f"Starting PARALLEL pipeline: query='{query}', num_papers={num_papers}")
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