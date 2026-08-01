"""
Parser Agent (Agent 2): PDF Section Extraction.

Extracts structured sections (introduction, methodology, results,
conclusion) from paper PDFs using font-size-based heading detection,
with a positional fallback when no headings can be reliably matched.
"""

from pathlib import Path
from collections import Counter
import re
import pdfplumber
from loguru import logger

# Canonical section names we look for, and the heading text variants
# that map to them. Matching is case-insensitive and substring-based
# against detected headings (not against the whole document).
SECTION_ALIASES = {
    "introduction": ["introduction"],
    "methodology": [
        "method", "methods", "methodology", "approach", "model",
        "architecture", "proposed method", "system", "our approach",
    ],
    "results": ["result", "results", "experiment", "experiments", "evaluation"],
    "conclusion": ["conclusion", "conclusions", "discussion"],
}

MAX_SECTION_CHARS = 3000  # cap per-section length fed to the LLM later


def _extract_words_in_reading_order(pdf_path: str) -> list[dict]:
    """
    Extract words from a PDF in correct reading order, handling both
    single- and two-column layouts.

    Parameters
    ----------
    pdf_path : str

    Returns
    -------
    list[dict]
        Words with `text`, `size`, and `page`.
    """
    all_words = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # x_tolerance controls how large a horizontal gap must be
            # before pdfplumber treats two adjacent characters as
            # separate words. The default (often ~3) is too loose for
            # justified academic PDF text, where inter-word spacing
            # can be quite narrow -- this was causing real bugs like
            # "tionchunksformoreaccuratevisual-textualmatching." being
            # returned as a single "word" with no internal spaces.
            # Verified by direct testing against real arXiv PDFs that
            # a tighter tolerance (1) correctly splits these cases
            # without over-splitting normal words.
            words = page.extract_words(extra_attrs=["size"], x_tolerance=1)
            if not words:
                continue

            page_width = page.width
            midpoint = page_width / 2

            # Detect whether this page is actually two-column by
            # checking if words cluster on both sides of the
            # midpoint, vs. being a single-column page (e.g. title
            # page, or a paper that just isn't two-column).
            left_count = sum(1 for w in words if w["x1"] < midpoint)
            right_count = sum(1 for w in words if w["x0"] > midpoint)
            is_two_column = left_count > 5 and right_count > 5

            if is_two_column:
                left_words = sorted(
                    [w for w in words if w["x0"] < midpoint],
                    key=lambda w: (w["top"], w["x0"]),
                )
                right_words = sorted(
                    [w for w in words if w["x0"] >= midpoint],
                    key=lambda w: (w["top"], w["x0"]),
                )
                ordered = left_words + right_words
            else:
                ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))

            for w in ordered:
                all_words.append({
                    "text": w["text"],
                    "size": round(w.get("size", 0), 1),
                    "page": page_num,
                })

    return all_words


def _detect_body_font_size(words: list[dict]) -> float:
    """Return the most common font size, used as a proxy for body text size."""

    sizes = [w["size"] for w in words if w["size"] > 0]
    if not sizes:
        return 0.0
    return Counter(sizes).most_common(1)[0][0]


def _group_into_lines(words: list[dict]) -> list[dict]:
    """Group words into lines based on font-size continuity."""
    lines = []
    current_line = []
    current_size = None

    for w in words:
        if current_size is not None and abs(w["size"] - current_size) > 0.3 and current_line:
            lines.append({
                "text": " ".join(x["text"] for x in current_line),
                "size": current_size,
            })
            current_line = []
        current_line.append(w)
        current_size = w["size"]

    if current_line:
        lines.append({
            "text": " ".join(x["text"] for x in current_line),
            "size": current_size,
        })

    return lines


def _match_heading(line_text: str) -> str | None:
    """Match a heading-sized line against known section aliases."""
    cleaned = re.sub(r"^[\d\.\s]+", "", line_text).strip().lower()
    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if cleaned == alias or cleaned.startswith(alias + " ") or cleaned.startswith(alias + ":"):
                return canonical
    return None


def extract_sections(pdf_path: str) -> tuple[dict, str, bool]:
    """
    Extract structured sections and full text from a paper PDF.

    Parameters
    ----------
    pdf_path : str

    Returns
    -------
    tuple[dict, str, bool]
        `(sections, full_text, low_confidence)`. `low_confidence` is
        True when no headings were detected and a positional fallback
        (text split into thirds) was used instead.
    """
    words = _extract_words_in_reading_order(pdf_path)
    if not words:
        # No extractable text at all -- likely a scanned/image PDF.
        # We don't attempt OCR here (out of scope for this agent);
        # the caller marks this paper as parse_failed.
        return {}, "", True

    full_text = " ".join(w["text"] for w in words)
    body_size = _detect_body_font_size(words)
    lines = _group_into_lines(words)

    sections = {}
    current_section = None
    buffer = []

    HEADING_SIZE_RATIO = 1.08  # heading must be at least 8% larger than body text

    for line in lines:
        is_heading_sized = body_size > 0 and line["size"] >= body_size * HEADING_SIZE_RATIO
        matched = _match_heading(line["text"]) if is_heading_sized and len(line["text"]) < 60 else None

        if matched:
            if current_section and buffer:
                sections[current_section] = " ".join(buffer)[:MAX_SECTION_CHARS]
            current_section = matched
            buffer = []
        elif current_section:
            buffer.append(line["text"])

    if current_section and buffer:
        sections[current_section] = " ".join(buffer)[:MAX_SECTION_CHARS]

    low_confidence = len(sections) == 0

    if low_confidence:
        # Fallback: no font-detected headings matched at all (common
        # on PDFs where heading/body font sizes are nearly identical,
        # or our column-split heuristic misread the layout). Rather
        # than returning nothing, take rough first/middle/last-third
        # slices of the full text as weak proxies. This is explicitly
        # lower quality and the low_confidence flag tells downstream
        # code (and eventually the UI) to treat it accordingly.
        logger.warning(f"No headings detected via font analysis for {pdf_path}, using fallback slicing")
        third = len(full_text) // 3
        sections = {
            "introduction": full_text[:third][:MAX_SECTION_CHARS],
            "methodology": full_text[third:2 * third][:MAX_SECTION_CHARS],
            "results": full_text[2 * third:][:MAX_SECTION_CHARS],
        }

    return sections, full_text, low_confidence


def parser_node(state: dict) -> dict:
    """
    LangGraph node wrapper around `extract_sections`.

    Parses each paper in `state["papers"]`; excludes papers with no
    extractable text from downstream processing.

    Returns
    -------
    dict
        Partial state update: `parsed_papers` and `errors`.
    """
    parsed_papers = []
    new_errors = []

    for paper in state.get("papers", []):
        try:
            sections, full_text, low_confidence = extract_sections(paper["pdf_path"])

            if not full_text:
                # Total extraction failure (likely scanned PDF) --
                # exclude from downstream processing rather than
                # passing empty content into the summarizer, which
                # would just hallucinate a summary from nothing.
                logger.warning(f"No text extracted from '{paper['title'][:60]}...' -- likely scanned/image PDF")
                new_errors.append(
                    f"Could not extract text from '{paper['title']}' (possibly a scanned PDF)"
                )
                continue

            parsed_papers.append({
                **paper,
                "full_text": full_text[:5000],
                "sections": sections,
                "parse_low_confidence": low_confidence,
            })

        except Exception as e:
            logger.warning(f"Parsing failed for '{paper['title'][:60]}...': {e}")
            new_errors.append(f"Parsing failed for '{paper['title']}': {e}")

    return {
        "parsed_papers": parsed_papers,
        "errors": new_errors,
    }