"""
Shared JSON extraction/validation utilities for LLM agents that need
structured output (summarizer, quality assessor, contradiction
detector).

Centralized here instead of duplicated per-agent, since the parsing
robustness fixes (bracket-counting instead of greedy regex, fallback
result shape) apply identically regardless of which agent is calling
it.
"""

import json
import re


def extract_json(raw_response: str) -> dict | None:
    """Extract and parse a JSON object from an LLM response.

    Tries direct parsing first, then falls back to locating the
    outermost balanced {...} block via bracket counting -- this
    correctly handles nested objects/arrays, unlike a greedy regex
    (r'\\{.*\\}') which can over-match across multiple JSON-like
    fragments or under-match on nested braces.
    """
    raw_response = raw_response.strip()

    # Strip markdown code fences if present, despite instructions not to use them
    raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
    raw_response = re.sub(r"\s*```$", "", raw_response)

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    start = raw_response.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(raw_response)):
        if raw_response[i] == "{":
            depth += 1
        elif raw_response[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_response[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def clamp_score(value, default=5, lo=1, hi=10):
    """Normalize a score field to an integer in [lo, hi]. Used for
    1-10 quality scores where the LLM might return floats, out-of-
    range values, or unparseable strings."""
    try:
        v = int(round(float(value)))
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def fallback_result(primary_field: str, reason: str, extra_fields: dict | None = None) -> dict:
    """Standard shape for 'the LLM response could not be parsed'
    results. Explicit and visible (fallback=True) rather than
    silently returning plausible-looking placeholder data that a
    user might mistake for a real result.
    """
    result = {
        primary_field: f"[Could not generate: {reason}]",
        "fallback": True,
    }
    if extra_fields:
        result.update(extra_fields)
    return result