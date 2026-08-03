"""
Agent 4: Quality Assessment Agent.

Evaluates each parsed paper along four dimensions - novelty, rigor,
clarity, and impact - using an LLM prompted to act as a critical
peer reviewer, grounding scores in the paper's abstract and extracted
methodology section rather than defaulting to generic praise.

Supports two execution modes:
    - `quality_assessment_node`: sequential assessment of all parsed
      papers in a single node call.
    - `assess_one_paper_node`: per-paper node intended for parallel
      fan-out execution, gated by a shared LLM concurrency semaphore.

Shared JSON parsing/validation utilities (`extract_json`, `clamp_score`,
`fallback_result`) live in `utils/json_extraction.py` and are also used
by the Summarizer agent, keeping validation logic consistent across
both LLM-backed agents.

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
    """
    Validate and normalize a raw parsed quality-assessment response.

    Clamps each score to a valid range, computes an overall average,
    and ensures list/string fields default to safe empty values if
    missing from the LLM's response.

    Parameters
    ----------
    parsed : dict
        Raw dict parsed from the LLM's JSON output.

    Returns
    -------
    dict
        Validated result containing `novelty`, `rigor`, `clarity`,
        `impact`, `overall` (average of the four, rounded to 1
        decimal), `quality_justification`, `strengths`, and
        `weaknesses`.
    """
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
    """
    Run quality assessment for a single paper via the LLM.

    Builds a grounded prompt from the paper's title, abstract, and
    extracted methodology section, then invokes the LLM and validates
    its JSON response. Retries once with a stricter formatting
    reminder if the first response fails to parse.

    Parameters
    ----------
    paper : dict
        Parsed paper dict, expected to contain `title`, `abstract`,
        and a `sections` dict (used to source `methodology`).

    Returns
    -------
    dict
        Validated quality assessment result from `_validate_quality`,
        or a fallback result (with a `fallback` flag) if the LLM call
        fails or its output cannot be parsed as valid JSON after
        retrying.
    """

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
    """
    LangGraph node: assess a single paper, for parallel fan-out execution.

    Intended to be invoked once per paper when the graph is running
    in parallel mode. Acquires `llm_semaphore` before calling the LLM
    to respect the configured `MAX_CONCURRENT_LLM_CALLS` limit.

    Parameters
    ----------
    state : dict
        Node-scoped state. Reads `paper`, a single parsed paper dict.

    Returns
    -------
    dict
        Partial state update:
        - `quality_scores` (dict): `{title: result}` for this paper.
        - `errors` (list[str]): populated if assessment fell back
          due to an LLM or parsing failure.
    """
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
    """
    LangGraph node: assess all parsed papers sequentially.

    Iterates over every paper in `state["parsed_papers"]`, running
    quality assessment one at a time. Used for the sequential workflow
    variant, as opposed to `assess_one_paper_node`'s parallel fan-out.

    Parameters
    ----------
    state : dict
        Current graph state. Reads `parsed_papers`, a list of parsed
        paper dicts.

    Returns
    -------
    dict
        Partial state update:
        - `quality_scores` (dict): `{title: result}` for every
          assessed paper.
        - `errors` (list[str]): one entry per paper whose assessment
          fell back due to an LLM or parsing failure.
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