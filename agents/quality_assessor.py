"""
Agent 4: Quality Assessment Agent.

Evaluates paper quality (novelty, rigor, clarity, impact) as a
simulated peer reviewer would. Kept separate from the Summarization
Agent -- see agents/summarizer.py docstring for the reasoning on why
these were split back apart after an earlier attempt to merge them.

The prompt here is deliberately framed as critical peer review (not
"explain what's good about this paper"), since a summarization-style
prompt tends to pull the model toward a descriptive, charitable tone
that produces inflated scores with little differentiation between
papers.

"""

from loguru import logger
from utils.llm_factory import get_llm
from utils.json_extraction import extract_json, clamp_score, fallback_result
from utils.concurrency import llm_semaphore

QUALITY_ASSESSMENT_PROMPT = """You are a senior, critical peer reviewer for a top-tier AI conference (NeurIPS/ICML level). Your job is to find real weaknesses, not just praise the paper.

Respond with ONLY a single valid JSON object, no preamble, no markdown code fences, no explanation before or after.

Required JSON structure:
{{
    "novelty": <integer 1-10>,
    "rigor": <integer 1-10>,
    "clarity": <integer 1-10>,
    "impact": <integer 1-10>,
    "quality_justification": "brief explanation grounding each score in specific evidence from the paper",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"]
}}

Scoring guide -- use the FULL range, do not default to 6-8 for everything:
- Novelty: 1-3 incremental/derivative, 4-6 moderate extension of existing work, 7-10 highly novel/groundbreaking
- Rigor: 1-3 major methodological flaws or missing baselines, 4-6 adequate but with gaps (e.g. no ablations), 7-10 exemplary (strong baselines, ablations, statistical validation)
- Clarity: 1-3 confusing/poorly structured, 4-6 understandable but could improve, 7-10 exceptionally clear
- Impact: 1-3 narrow/incremental applicability, 4-6 useful for a specific subfield, 7-10 broadly transformative potential

Important: if a section below says "Not available in extracted text", explicitly lower your confidence in rigor/clarity scores accordingly and say so in the justification, rather than scoring as if you had seen the full methodology.

---
Paper Title: {title}

Abstract: {abstract}

Methodology Section: {methodology}
---

Respond with ONLY the JSON object."""


def _validate_quality(parsed: dict) -> dict:
    novelty = clamp_score(parsed.get("novelty"))
    rigor = clamp_score(parsed.get("rigor"))
    clarity = clamp_score(parsed.get("clarity"))
    impact = clamp_score(parsed.get("impact"))

    return {
        "novelty": novelty,
        "rigor": rigor,
        "clarity": clarity,
        "impact": impact,
        "overall": round((novelty + rigor + clarity + impact) / 4, 1),
        "quality_justification": str(parsed.get("quality_justification", "")),
        "strengths": parsed.get("strengths", []) or [],
        "weaknesses": parsed.get("weaknesses", []) or [],
    }


def assess(paper: dict) -> dict:
    """Run quality assessment for one paper."""
    llm = get_llm("quality_assessor")
    sections = paper.get("sections", {})

    prompt = QUALITY_ASSESSMENT_PROMPT.format(
        title=paper.get("title", "Unknown"),
        abstract=(paper.get("abstract", "") or "Not available in extracted text")[:800],
        methodology=sections.get("methodology", "Not available in extracted text")[:1500],
    )

    extra = {
        "novelty": None, "rigor": None, "clarity": None, "impact": None,
        "overall": None, "quality_justification": "", "strengths": [], "weaknesses": [],
    }

    for attempt in range(2):
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            logger.warning(f"Quality assessor LLM call failed for '{paper.get('title', '')[:50]}...': {e}")
            return fallback_result("quality_justification", f"LLM invocation error: {e}", extra)

        parsed = extract_json(response)
        if parsed is not None:
            return _validate_quality(parsed)

        logger.warning(f"Quality JSON parse failed (attempt {attempt + 1}/2) for '{paper.get('title', '')[:50]}...'")
        prompt = prompt + "\n\nReminder: respond with ONLY valid JSON, nothing else."

    logger.error(f"Could not parse quality JSON after retries for '{paper.get('title', '')[:50]}...'")
    return fallback_result("quality_justification", "LLM did not return valid JSON after 2 attempts", extra)


def assess_one_paper_node(state:dict) -> dict:
    """Per-paper node for parallel Send-based fan-out."""
    paper = state["paper"]
    title = paper["title"]
    logger.info(f"[parallel] Assessing quality: {title[:60]}... (waiting for LLM slot)")
    
    with llm_semaphore:
        logger.info(f"[parallel] Got LLM slot, assessing: {title[:60]}...")
        result = assess(paper)

    new_errors = []
    if result.get("fallback"):
        new_errors.append(f"Quality assessment failed for '{title}'")

    return {
        "quality_scores": {title: result},
        "errors": new_errors,
    }


def quality_assessment_node(state: dict) -> dict:
    """LangGraph node wrapper. Sequential for now -- parallel fan-out
    comes later when we wire up the full graph.

    Returns a partial state update, not the mutated whole state --
    see agents/ingestion.py's ingestion_node docstring for why this
    matters with LangGraph's Annotated/operator.add reducer fields.
    """
    new_errors = []
    scores = {}

    for paper in state.get("parsed_papers", []):
        title = paper["title"]
        logger.info(f"Assessing quality: {title[:60]}...")
        result = assess(paper)

        if result.get("fallback"):
            new_errors.append(f"Quality assessment failed for '{title}'")

        scores[title] = result

    return {
        "quality_scores": scores,
        "errors": new_errors,
    }