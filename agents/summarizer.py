"""
Agent 3: Summarization Agent.

Generates a structured, faithful summary of a paper's contribution,
methodology, findings, and limitations.

"""

from loguru import logger
from utils.llm_factory import get_llm
from utils.json_extraction import extract_json, fallback_result

SUMMARIZATION_PROMPT = """You are an expert research summarizer. Analyze this paper and create a structured summary.

Respond with ONLY a single valid JSON object, no preamble, no markdown code fences, no explanation before or after.

Required JSON structure:
{{
    "main_contribution": "2-3 sentences capturing the core innovation",
    "methodology_summary": ["bullet point 1", "bullet point 2", "bullet point 3"],
    "key_findings": ["finding 1 with metrics if available", "finding 2", "finding 3"],
    "limitations": "1-2 sentences on weaknesses or constraints"
}}

Important: if a section below says "Not available in extracted text", do not invent details for it -- work from what IS available and note the gap instead of fabricating content.

Be technical, precise, and concise. Focus on reproducible facts.

---
Paper Title: {title}

Abstract: {abstract}

Methodology Section: {methodology}

Results Section: {results}
---

Respond with ONLY the JSON object."""


def _validate_summary(parsed: dict) -> dict:
    return {
        "main_contribution": str(parsed.get("main_contribution", "Not available")),
        "methodology_summary": parsed.get("methodology_summary", []) or [],
        "key_findings": parsed.get("key_findings", []) or [],
        "limitations": str(parsed.get("limitations", "Not available")),
    }


def summarize(paper: dict) -> dict:
    """Run summarization for one paper."""
    llm = get_llm("summarizer")
    sections = paper.get("sections", {})

    prompt = SUMMARIZATION_PROMPT.format(
        title=paper.get("title", "Unknown"),
        abstract=(paper.get("abstract", "") or "Not available in extracted text")[:800],
        methodology=sections.get("methodology", "Not available in extracted text")[:1500],
        results=sections.get("results", "Not available in extracted text")[:1500],
    )

    extra = {"methodology_summary": [], "key_findings": [], "limitations": ""}

    for attempt in range(2):
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            logger.warning(f"Summarizer LLM call failed for '{paper.get('title', '')[:50]}...': {e}")
            return fallback_result("main_contribution", f"LLM invocation error: {e}", extra)

        parsed = extract_json(response)
        if parsed is not None:
            return _validate_summary(parsed)

        logger.warning(f"Summary JSON parse failed (attempt {attempt + 1}/2) for '{paper.get('title', '')[:50]}...'")
        prompt = prompt + "\n\nReminder: respond with ONLY valid JSON, nothing else."

    logger.error(f"Could not parse summary JSON after retries for '{paper.get('title', '')[:50]}...'")
    return fallback_result("main_contribution", "LLM did not return valid JSON after 2 attempts", extra)


def summarization_node(state: dict) -> dict:
    """LangGraph node wrapper. Sequential for now -- parallel fan-out
    comes later when we wire up the full graph."""
    state.setdefault("errors", [])
    summaries = {}

    for paper in state.get("parsed_papers", []):
        title = paper["title"]
        logger.info(f"Summarizing: {title[:60]}...")
        result = summarize(paper)

        if result.get("fallback"):
            state["errors"].append(f"Summarization failed for '{title}'")

        summaries[title] = result

    state["summaries"] = summaries
    return state