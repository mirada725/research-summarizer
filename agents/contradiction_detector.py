"""
Agent 5: Contradiction Detector.

Compares paper summaries pairwise to flag potential contradictions:
1. Embed each summary with a sentence-transformer
2. Compute pairwise cosine similarity
3. For pairs ABOVE a similarity floor (i.e. same general topic), ask
   the LLM to verify whether they actually contradict

IMPORTANT -- THIS LOGIC IS INVERTED FROM THE ORIGINAL PLAN, BASED ON
REAL MEASUREMENT, NOT GUESSWORK:

The original plan assumed "low similarity = potential contradiction"
and used a `similarity < 0.4` cutoff. We tested this empirically
(utils/calibrate_similarity_threshold.py) against realistic summary
pairs and found the opposite:

    same_topic_agreeing       -> 0.763
    same_topic_contradicting  -> 0.890   <- HIGHER than agreeing!
    unrelated_topics          -> 0.016

A genuine contradiction (e.g. "removing NSP improves performance" vs
"removing NSP degrades performance") uses nearly identical vocabulary
to state the opposite claim, so it embeds as HIGHLY similar -- often
more similar than two papers that loosely agree but use different
terminology. This means cosine similarity cannot distinguish
agreement from contradiction; it can only reliably tell you whether
two papers are about the same narrow topic at all (the unrelated case
dropped to 0.016, a clean, unambiguous signal).

Given that, the embedding step here is repurposed as what it's
actually good for: a cheap pre-filter that skips LLM verification for
pairs that are CLEARLY unrelated topics (saving real time/Ollama load
on an 8GB laptop), while sending every same-topic-or-better pair to
the LLM, since that's where real contradictions actually hide. This
is a low floor, not a tight threshold -- the point is to skip only
the obviously-irrelevant pairs, not to pre-judge agreement.

If you re-run calibrate_similarity_threshold.py against your own
real paper data and see different patterns, adjust
SIMILARITY_FLOOR accordingly -- don't assume the numbers above
generalize perfectly to every paper domain.

Other fixes applied vs. the original plan:
- Guards against n<2 papers (can't detect contradictions with one
  paper -- original plan doesn't check this and would silently do
  nothing useful or error on an empty pair list).
- Reuses the shared, robust JSON extraction from
  utils/json_extraction.py instead of trusting raw LLM output shape.
- The sentence-transformer model is loaded once per node call, not
  once per pair comparison.
"""

from itertools import combinations
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.llm_factory import get_llm
from utils.json_extraction import extract_json, fallback_result

# Pairs with similarity BELOW this are skipped entirely (clearly
# unrelated topics -- measured at ~0.016 for genuinely unrelated
# papers, vs. 0.7-0.9 for same-topic papers regardless of whether
# they agree or contradict). This is a low floor to filter out
# obviously-irrelevant pairs and save LLM calls, NOT a contradiction
# signal -- see module docstring for why a tight/inverted threshold
# would miss real contradictions.
SIMILARITY_FLOOR = 0.3

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CONTRADICTION_VERIFICATION_PROMPT = """You are analyzing two research paper summaries to determine if they genuinely contradict each other.

A real contradiction means the papers make incompatible factual claims about the same phenomenon (e.g. one says X improves performance, the other says X degrades performance, under comparable conditions).

This is NOT a contradiction if:
- The papers simply address different topics or problems
- They use different methods/datasets and aren't claiming the same thing
- One is a strict superset/extension of the other's claims

Respond with ONLY a single valid JSON object, no preamble, no markdown fences.

Required JSON structure:
{{
    "contradicts": true or false,
    "conflict_type": "direct" or "methodological" or "none",
    "severity": "low" or "medium" or "high",
    "explanation": "1-2 sentences on why they do or don't contradict, citing the specific claims"
}}

---
Paper 1: {title1}
Summary: {summary1}

Paper 2: {title2}
Summary: {summary2}
---

Respond with ONLY the JSON object."""


def _summary_to_text(summary: dict) -> str:
    """Flatten a structured summary dict into plain text for
    embedding. Uses main_contribution + key_findings, since those
    carry the actual claims most relevant to contradiction detection
    (methodology/limitations are less likely to directly conflict)."""
    if summary.get("fallback"):
        return ""
    parts = [summary.get("main_contribution", "")]
    parts.extend(summary.get("key_findings", []))
    return " ".join(p for p in parts if p)


def _verify_contradiction(title1: str, summary1: str, title2: str, summary2: str) -> dict:
    llm = get_llm("contradiction_detector")
    prompt = CONTRADICTION_VERIFICATION_PROMPT.format(
        title1=title1, summary1=summary1[:600],
        title2=title2, summary2=summary2[:600],
    )

    extra = {"contradicts": False, "conflict_type": "none", "severity": "low", "explanation": ""}

    for attempt in range(2):
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            logger.warning(f"Contradiction LLM call failed for '{title1[:30]}' vs '{title2[:30]}': {e}")
            return fallback_result("explanation", f"LLM invocation error: {e}", extra)

        parsed = extract_json(response)
        if parsed is not None:
            return {
                "contradicts": bool(parsed.get("contradicts", False)),
                "conflict_type": str(parsed.get("conflict_type", "none")),
                "severity": str(parsed.get("severity", "low")),
                "explanation": str(parsed.get("explanation", "")),
            }

        prompt = prompt + "\n\nReminder: respond with ONLY valid JSON, nothing else."

    return fallback_result("explanation", "LLM did not return valid JSON after 2 attempts", extra)


def detect_contradictions(
    summaries: dict[str, dict],
    similarity_floor: float = SIMILARITY_FLOOR,
) -> list[dict]:
    """Compare all paper summary pairs and return flagged contradictions.

    summaries: {title: summary_dict} as produced by agents/summarizer.py

    Pairs below similarity_floor are skipped as clearly-unrelated
    topics. Everything else is sent to the LLM for verification,
    since embedding similarity cannot distinguish agreement from
    contradiction among same-topic papers (see module docstring).
    """
    titles = list(summaries.keys())

    if len(titles) < 2:
        logger.info(f"Only {len(titles)} paper(s) available; skipping contradiction detection (needs >= 2).")
        return []

    texts = {t: _summary_to_text(summaries[t]) for t in titles}

    usable_titles = [t for t in titles if texts[t].strip()]
    if len(usable_titles) < 2:
        logger.warning("Fewer than 2 papers have usable summaries; skipping contradiction detection.")
        return []

    logger.info(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = encoder.encode([texts[t] for t in usable_titles])
    sim_matrix = cosine_similarity(embeddings)

    contradictions = []
    checked_count = 0
    skipped_count = 0

    for i, j in combinations(range(len(usable_titles)), 2):
        similarity = float(sim_matrix[i][j])
        title1, title2 = usable_titles[i], usable_titles[j]

        if similarity < similarity_floor:
            logger.info(
                f"Skipping clearly-unrelated pair (similarity={similarity:.3f}): "
                f"'{title1[:40]}...' / '{title2[:40]}...'"
            )
            skipped_count += 1
            continue

        logger.info(
            f"Same-topic pair (similarity={similarity:.3f}), verifying with LLM: "
            f"'{title1[:40]}...' vs '{title2[:40]}...'"
        )
        checked_count += 1
        verification = _verify_contradiction(title1, texts[title1], title2, texts[title2])

        if verification.get("contradicts"):
            contradictions.append({
                "paper1": title1,
                "paper2": title2,
                "similarity": round(similarity, 3),
                **{k: v for k, v in verification.items() if k != "contradicts"},
            })

    logger.info(f"Contradiction check complete: {checked_count} pairs verified by LLM, {skipped_count} skipped as unrelated")
    return contradictions


def contradiction_detector_node(state: dict) -> dict:
    """LangGraph node wrapper."""
    state.setdefault("errors", [])
    summaries = state.get("summaries", {})

    contradictions = detect_contradictions(summaries)
    state["contradictions"] = contradictions

    logger.info(f"Found {len(contradictions)} contradiction(s) among {len(summaries)} paper(s)")
    return state