"""
Agent 6: Synthesis Agent.

Combines summaries, quality scores, and contradictions across all
papers into one coherent literature review report.

Fixes applied vs. the original plan:
- Papers whose summary or quality assessment failed (fallback=True,
  built in agents/summarizer.py and agents/quality_assessor.py) are
  explicitly excluded from the synthesis input and listed separately
  as "excluded due to processing errors" in the report, rather than
  silently feeding placeholder text like "[Could not generate: ...]"
  into the LLM as if it were real paper content.
- Quality scores actually drive structure: papers are sorted by
  overall score and the prompt explicitly asks the LLM to lead with
  the highest-scoring papers in "Key Innovations" -- the original
  plan computed quality scores but didn't actually use them to shape
  the synthesis beyond decoration.
- Context length is checked against the configured num_ctx before
  sending, with a warning logged if we're likely to truncate, rather
  than silently letting Ollama cut off mid-prompt.
- "No contradictions detected" is phrased as a positive finding in
  the prompt instructions, not dead placeholder text.
"""

from loguru import logger
from utils.llm_factory import get_llm
from utils.model_config import get_model_config

SYNTHESIS_PROMPT = """You are writing a comprehensive literature review for a research audience.

You have analyzed {num_papers} paper(s) on "{query}".
{excluded_note}
Create a structured review with these sections, using markdown formatting:

## Overview
One paragraph: number of papers analyzed, general scope, average quality score ({avg_quality}/10).

## Common Themes
3-5 themes or patterns that appear across multiple papers. Note where papers build on or relate to each other.

## Key Innovations
Highlight the top-scoring papers first (they are listed in descending quality order below -- lead with these). Include their quality scores and cite specific findings/metrics from their summaries. Keep methodology (how the work was done) and findings (what the results were) clearly distinct -- do not blend a result/metric into a methodology description.

## Contradictions & Debates
{contradictions_instruction}

## Research Gaps
Based on the limitations noted across papers, identify under-explored areas or open questions.

## Conclusion
Synthesize the overall state of this research area and suggest future directions.

Be scholarly but accessible. Ground every claim in the paper data provided below -- do not invent papers, metrics, or findings not present in the input.

---
PAPERS (sorted by quality score, highest first):

{papers_block}

---
CONTRADICTIONS DETECTED:

{contradictions_block}
---

Write the full literature review now."""


def _format_paper_block(title: str, summary: dict, quality: dict) -> str:
    findings = "\n".join(f"  - {f}" for f in summary.get("key_findings", [])) or "  - Not available"
    methodology = "\n".join(f"  - {m}" for m in summary.get("methodology_summary", [])) or "  - Not available"
    strengths = ", ".join(quality.get("strengths", [])) or "Not assessed"
    weaknesses = ", ".join(quality.get("weaknesses", [])) or "Not assessed"

    return f"""### {title}
Quality Score: {quality.get('overall', 'N/A')}/10 (Novelty: {quality.get('novelty', 'N/A')}, Rigor: {quality.get('rigor', 'N/A')}, Clarity: {quality.get('clarity', 'N/A')}, Impact: {quality.get('impact', 'N/A')})

Main Contribution: {summary.get('main_contribution', 'Not available')}

Methodology (how the work was done):
{methodology}

Key Findings (results/outcomes, NOT methodology):
{findings}

Limitations: {summary.get('limitations', 'Not available')}

Strengths: {strengths}
Weaknesses: {weaknesses}
"""


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token for English text. Good
    enough to warn about likely truncation, not meant to be exact."""
    return len(text) // 4


def synthesize(
    query: str,
    summaries: dict[str, dict],
    quality_scores: dict[str, dict],
    contradictions: list[dict],
) -> dict:
    """Generate the final literature review.

    Returns a dict with 'report' (the markdown text) and 'excluded_papers'
    (titles that were dropped due to fallback/failed processing), so the
    caller/UI can show both.
    """
    usable_titles = []
    excluded_titles = []

    for title, summary in summaries.items():
        quality = quality_scores.get(title, {})
        if summary.get("fallback") or quality.get("fallback"):
            excluded_titles.append(title)
        else:
            usable_titles.append(title)

    if not usable_titles:
        logger.error("No papers have usable summary+quality data; cannot synthesize a report.")
        return {
            "report": "# Literature Review\n\nNo papers could be successfully processed. Please check the errors log and try again.",
            "excluded_papers": excluded_titles,
        }

    # Sort by quality score, highest first -- this is what actually
    # makes "Key Innovations" lead with the best papers instead of
    # just listing them in arbitrary/ingestion order.
    usable_titles.sort(
        key=lambda t: quality_scores[t].get("overall") or 0,
        reverse=True,
    )

    papers_block = "\n\n".join(
        _format_paper_block(t, summaries[t], quality_scores[t]) for t in usable_titles
    )

    avg_quality = round(
        sum(quality_scores[t].get("overall", 0) for t in usable_titles) / len(usable_titles), 1
    )

    if contradictions:
        contradictions_block = "\n\n".join(
            f"- **{c['paper1']}** vs **{c['paper2']}** "
            f"(similarity: {c.get('similarity', 'N/A')}, severity: {c.get('severity', 'N/A')})\n"
            f"  {c.get('explanation', '')}"
            for c in contradictions
        )
        contradictions_instruction = "List and analyze each contradiction below, discussing possible reasons for the disagreement (different datasets, methods, or scale) and noting any apparent consensus."
    else:
        contradictions_block = "None detected among the analyzed papers."
        contradictions_instruction = "No contradictions were detected among these papers -- note this as a positive finding (the papers are broadly consistent), don't treat it as a gap."

    excluded_note = ""
    if excluded_titles:
        excluded_note = (
            f"\nNote: {len(excluded_titles)} paper(s) could not be fully processed and are "
            f"excluded from this analysis: {', '.join(excluded_titles)}.\n"
        )

    prompt = SYNTHESIS_PROMPT.format(
        num_papers=len(usable_titles),
        query=query,
        avg_quality=avg_quality,
        excluded_note=excluded_note,
        papers_block=papers_block,
        contradictions_block=contradictions_block,
        contradictions_instruction=contradictions_instruction,
    )

    cfg = get_model_config("synthesizer")
    estimated_tokens = _estimate_tokens(prompt)
    if estimated_tokens > cfg.num_ctx * 0.8:
        logger.warning(
            f"Synthesis prompt (~{estimated_tokens} estimated tokens) is approaching "
            f"the configured context window ({cfg.num_ctx}). Output may be truncated "
            f"or lose early context. Consider reducing num_papers or raising num_ctx "
            f"in utils/model_config.py for the synthesizer agent."
        )

    llm = get_llm("synthesizer")

    try:
        report = llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Synthesis LLM call failed: {e}")
        report = (
            f"# Literature Review: {query}\n\n"
            f"Report generation failed due to an error: {e}\n\n"
            f"Individual paper summaries and quality scores were still generated "
            f"successfully and can be reviewed separately."
        )

    return {"report": report, "excluded_papers": excluded_titles}


def synthesis_node(state: dict) -> dict:
    """LangGraph node wrapper."""
    state.setdefault("errors", [])

    result = synthesize(
        query=state.get("query", "research topic"),
        summaries=state.get("summaries", {}),
        quality_scores=state.get("quality_scores", {}),
        contradictions=state.get("contradictions", []),
    )

    state["final_report"] = result["report"]
    if result["excluded_papers"]:
        state["errors"].append(
            f"{len(result['excluded_papers'])} paper(s) excluded from synthesis due to processing failures: "
            f"{', '.join(result['excluded_papers'])}"
        )

    logger.info("Synthesis complete.")
    return state