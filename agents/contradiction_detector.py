"""
Agent 5: Contradiction Detector.

Compares paper summaries pairwise to flag potential contradictions:
1. Embed each summary with a sentence-transformer
2. Compute pairwise cosine similarity
3. For pairs below a similarity threshold, ask the LLM to verify
   whether they actually contradict (similarity alone can't tell you
   THIS -- it just flags "these are talking about different things,"
   not "these disagree")

IMPORTANT CAVEAT ON THE THRESHOLD (read before trusting the default):
The original plan used a hardcoded `similarity < 0.4` cutoff with no
justification. I could not empirically verify embedding behavior in
my own sandbox (no network access to huggingface.co to download the
model), so the default here is a reasoned estimate, not a measured
value -- and it may well be WRONG for your actual data. Specifically:
two papers that genuinely CONTRADICT each other (e.g. "removing NSP
improves performance" vs "removing NSP degrades performance") usually
still share heavy topical vocabulary ("NSP", "BERT", "pretraining",
"performance"), which can keep cosine similarity surprisingly HIGH --
often in the 0.5-0.7 range, not below 0.4. A low threshold tuned for
"these are about completely different topics" will likely MISS real
contradictions between same-topic papers, which is exactly the
interesting case you want this agent to catch.

Use utils/calibrate_similarity_threshold.py (run on your machine,
where the model can actually download) to check real similarity
scores on your own paper summaries before trusting this default in
production. The threshold is a config value specifically so you can
adjust it once you have real numbers, without touching this file.

Other fixes applied vs. the original plan:
- Guards against n<2 papers (can't detect contradictions with one
  paper -- original plan doesn't check this and would silently do
  nothing useful or error on an empty pair list).
- Reuses the shared, robust JSON extraction from
  utils/json_extraction.py instead of trusting raw LLM output shape.
- The sentence-transformer model is loaded once per node call, not
  once per pair comparison (the original plan's class-based version
  did this correctly already; flagging here since the original
  doc's loose Step-by-step code in another section did not).
"""

from itertools import combinations
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.llm_factory import get_llm
from utils.json_extraction import extract_json, fallback_result

# See the module docstring above -- this is a reasoned estimate, not
# a measured value. Calibrate against your real data before trusting
# it. Pairs with similarity BELOW this go to the LLM for verification.
SIMILARITY_THRESHOLD = 0.6

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
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Compare all paper summary pairs and return flagged contradictions.

    summaries: {title: summary_dict} as produced by agents/summarizer.py
    """
    titles = list(summaries.keys())

    if len(titles) < 2:
        # Original plan doesn't guard against this -- with 0 or 1
        # papers there's nothing to compare, so return early rather
        # than attempting pairwise comparisons on an empty/singleton
        # list (which would silently produce nothing useful anyway,
        # better to be explicit about why).
        logger.info(f"Only {len(titles)} paper(s) available; skipping contradiction detection (needs >= 2).")
        return []

    texts = {t: _summary_to_text(summaries[t]) for t in titles}

    # Papers whose summarization failed entirely have no usable text
    # to embed -- exclude them rather than embedding an empty string,
    # which would produce a meaningless embedding that could falsely
    # "match" or "mismatch" everything.
    usable_titles = [t for t in titles if texts[t].strip()]
    if len(usable_titles) < 2:
        logger.warning("Fewer than 2 papers have usable summaries; skipping contradiction detection.")
        return []

    logger.info(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = encoder.encode([texts[t] for t in usable_titles])
    sim_matrix = cosine_similarity(embeddings)

    contradictions = []

    for i, j in combinations(range(len(usable_titles)), 2):
        similarity = float(sim_matrix[i][j])
        if similarity < threshold:
            title1, title2 = usable_titles[i], usable_titles[j]
            logger.info(
                f"Low similarity ({similarity:.2f}) between "
                f"'{title1[:40]}...' and '{title2[:40]}...' -- verifying with LLM"
            )
            verification = _verify_contradiction(title1, texts[title1], title2, texts[title2])

            if verification.get("contradicts"):
                contradictions.append({
                    "paper1": title1,
                    "paper2": title2,
                    "similarity": round(similarity, 3),
                    **{k: v for k, v in verification.items() if k != "contradicts"},
                })

    return contradictions


def contradiction_detector_node(state: dict) -> dict:
    """LangGraph node wrapper."""
    state.setdefault("errors", [])
    summaries = state.get("summaries", {})

    contradictions = detect_contradictions(summaries)
    state["contradictions"] = contradictions

    logger.info(f"Found {len(contradictions)} contradiction(s) among {len(summaries)} paper(s)")
    return state