"""
Streamlit UI for the Agentic Research Paper Summarizer.

"""

import json
import streamlit as st
from loguru import logger

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="🔬",
    layout="wide",
)

# ── Friendly display names for each LangGraph node ─────────────────
NODE_LABELS = {
    "ingest":                "📥 Fetching papers from arXiv...",
    "parse":                 "📄 Parsing PDFs...",
    "summarize":             "✍️  Generating summaries (sequential)...",
    "summarize_one":         "✍️  Summarizing papers (parallel)...",
    "post_summarize":        "✍️  Summaries complete, preparing quality assessment...",
    "assess_quality":        "🔍 Assessing quality (sequential)...",
    "assess_one":            "🔍 Assessing paper quality (parallel)...",
    "detect_contradictions": "⚡ Detecting contradictions...",
    "synthesize":            "📝 Synthesizing literature review...",
    "early_exit":            "⚠️  Pipeline exited early (see errors below).",
}

PIPELINE_STAGES = [
    "ingest", "parse", "summarize / summarize_one",
    "assess_quality / assess_one", "detect_contradictions", "synthesize",
]


# ── Session state initialization ────────────────────────────────────
def _init_state():
    defaults = {
        "result": None,
        "running": False,
        "sequential_app": None,
        "parallel_app": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_app(use_parallel: bool):
    """Compile and cache the workflow graph in session state so it
    isn't recompiled on every Streamlit rerun/button click."""
    if use_parallel:
        if st.session_state.parallel_app is None:
            from graph.workflow_parallel import create_workflow
            st.session_state.parallel_app = create_workflow()
        return st.session_state.parallel_app
    else:
        if st.session_state.sequential_app is None:
            from graph.workflow import create_workflow
            st.session_state.sequential_app = create_workflow()
        return st.session_state.sequential_app


def _run_pipeline(query: str, num_papers: int, use_parallel: bool):
    """Run the pipeline with real-time progress via st.status() and
    LangGraph's stream_mode='updates', which yields a dict per
    completed node containing {node_name: partial_state_update}."""
    from graph.state import ResearchState

    app = _get_app(use_parallel)

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

    completed_nodes = set()
    final_state = dict(initial_state)

    with st.status("Running pipeline...", expanded=True) as status:
        progress = st.progress(0)

        try:
            for chunk in app.stream(initial_state, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    if node_name in ("__start__", "__end__"):
                        continue
                    if not isinstance(node_update, dict):
                        continue

                    label = NODE_LABELS.get(node_name, f"Running {node_name}...")
                    st.write(label)
                    completed_nodes.add(node_name)

                    # Merge partial updates into our running state view
                    # (same logic as LangGraph's own reducer, simplified)
                    for k, v in node_update.items():
                        if isinstance(v, list) and isinstance(final_state.get(k), list):
                            final_state[k] = final_state[k] + v
                        elif isinstance(v, dict) and isinstance(final_state.get(k), dict):
                            final_state[k] = {**final_state.get(k, {}), **v}
                        else:
                            final_state[k] = v

                    # Rough progress: count distinct non-parallel node
                    # completions as a fraction of expected stages
                    stage_key = node_name.split("_one")[0]
                    pct = min(
                        int(len({n.split("_one")[0] for n in completed_nodes}) / len(PIPELINE_STAGES) * 90),
                        90
                    )
                    progress.progress(pct)

            progress.progress(100)
            status.update(label="✅ Pipeline complete!", state="complete", expanded=False)

        except Exception as e:
            status.update(label=f"❌ Pipeline failed: {e}", state="error")
            logger.exception("Pipeline error")
            st.error(str(e))
            return None

    return final_state


# ── Main UI ─────────────────────────────────────────────────────────
def main():
    _init_state()

    st.title("🔬 Agentic Research Paper Summarizer")
    st.caption("Powered by LangGraph + Ollama — 100% local, 100% free")

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        query = st.text_input(
            "Research Query",
            value="transformer models in NLP",
            help="Searches arXiv for the most relevant recent papers on this topic.",
        )

        num_papers = st.slider(
            "Number of Papers",
            min_value=2,
            max_value=10,
            value=3,
            help="More papers = longer run time. Start with 2-3 on an 8GB machine.",
        )

        use_parallel = st.toggle(
            "Parallel processing",
            value=False,
            help=(
                "Parallel mode processes papers concurrently (faster for 4+ papers). "
                "On an 8GB laptop, set MAX_CONCURRENT_LLM_CALLS=1 in .env first "
                "to avoid RAM exhaustion."
            ),
        )

        st.divider()
        st.caption(
            f"**Mode:** {'Parallel' if use_parallel else 'Sequential'}  \n"
            f"**Model:** llama3.1:8b (configurable in utils/model_config.py)"
        )
        st.divider()

        run_clicked = st.button(
            "🚀 Generate Literature Review",
            type="primary",
            disabled=st.session_state.running,
            use_container_width=True,
        )

    # ── Run pipeline ─────────────────────────────────────────────────
    if run_clicked and not st.session_state.running:
        if not query.strip():
            st.sidebar.error("Please enter a research query.")
        else:
            st.session_state.running = True
            st.session_state.result = None

            result = _run_pipeline(query.strip(), num_papers, use_parallel)
            st.session_state.result = result
            st.session_state.running = False
            st.rerun()

    # ── Results ──────────────────────────────────────────────────────
    result = st.session_state.result

    if result is None:
        st.info(
            "👈 Enter a research query and click **Generate Literature Review** to start.\n\n"
            "The pipeline will fetch papers from arXiv, parse their PDFs, "
            "generate summaries, assess quality, detect contradictions, "
            "and synthesize a literature review — fully locally."
        )
        return

    # ── Errors banner ────────────────────────────────────────────────
    errors = result.get("errors", [])
    if errors:
        with st.expander(f"⚠️ {len(errors)} issue(s) encountered during run", expanded=False):
            for e in errors:
                st.warning(e)

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Literature Review",
        "📊 Quality Scores",
        "⚡ Contradictions",
        "🔍 Raw Data",
    ])

    with tab1:
        report = result.get("final_report", "")
        if report:
            st.markdown(report)
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download as Markdown",
                    data=report,
                    file_name="literature_review.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "📦 Download Raw Data (JSON)",
                    data=json.dumps(result, indent=2, default=str),
                    file_name="research_data.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.error("No report was generated. Check the issues panel above.")

    with tab2:
        quality_scores = result.get("quality_scores", {})
        if not quality_scores:
            st.info("No quality scores available.")
        else:
            for title, scores in quality_scores.items():
                if scores.get("fallback"):
                    st.warning(f"**{title}** — quality assessment failed")
                    continue

                with st.expander(f"**{title[:90]}**  —  Overall: {scores.get('overall', 'N/A')}/10"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Novelty", f"{scores.get('novelty', 'N/A')}/10")
                    col2.metric("Rigor", f"{scores.get('rigor', 'N/A')}/10")
                    col3.metric("Clarity", f"{scores.get('clarity', 'N/A')}/10")
                    col4.metric("Impact", f"{scores.get('impact', 'N/A')}/10")

                    if scores.get("quality_justification"):
                        st.info(scores["quality_justification"])

                    scol1, scol2 = st.columns(2)
                    with scol1:
                        st.markdown("**Strengths**")
                        for s in scores.get("strengths", []):
                            st.markdown(f"- {s}")
                    with scol2:
                        st.markdown("**Weaknesses**")
                        for w in scores.get("weaknesses", []):
                            st.markdown(f"- {w}")

    with tab3:
        contradictions = result.get("contradictions", [])
        if not contradictions:
            st.success(
                "✅ No contradictions detected among the analyzed papers. "
                "The papers are broadly consistent with each other."
            )
        else:
            for c in contradictions:
                severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    c.get("severity", "low"), "🟡"
                )
                with st.expander(
                    f"{severity_color} **{c['paper1'][:60]}** vs **{c['paper2'][:60]}**"
                ):
                    col1, col2 = st.columns(2)
                    col1.metric("Similarity Score", f"{c.get('similarity', 'N/A'):.3f}")
                    col2.metric("Severity", c.get("severity", "N/A").title())
                    st.markdown(f"**Conflict type:** {c.get('conflict_type', 'N/A')}")
                    st.markdown(f"**Explanation:** {c.get('explanation', 'N/A')}")

    with tab4:
        st.json(result)


if __name__ == "__main__":
    main()