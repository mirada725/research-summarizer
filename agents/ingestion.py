"""
Agent 1: Ingestion Agent.

Fetches papers from arXiv and downloads their PDFs.

Fixes applied vs. the original plan:
- Uses arxiv.Client().results(search) instead of the deprecated
  search.results() method (removed in arxiv>=2.x).
- IMPORTANT: arxiv>=4.0.0 removed Result.download_pdf() entirely --
  the library is now metadata/search-only. We fetch the actual PDF
  bytes ourselves via `requests` using result.pdf_url. This was
  discovered by testing against the real installed version (4.0.0),
  not assumed from older docs -- worth remembering that this library
  has changed its API surface significantly across majors, so if you
  ever bump arxiv's version again, re-verify this against a real run
  rather than trusting cached knowledge of the API.
- Respects arXiv's rate-limit guidance (3 seconds between requests)
  via the Client's own delay_seconds for search/metadata calls, AND
  a manual delay before each PDF download too, since downloads hit
  arxiv.org directly (a separate rate-limit surface from the API).
- Wraps each paper's download in a try/except so one bad PDF
  (network blip, malformed file, 404) doesn't kill the whole batch --
  it's recorded as a failure and the rest continue.
- Skips re-downloading PDFs that are already cached on disk.
"""

import time
from pathlib import Path
import arxiv
import requests
from loguru import logger

CACHE_DIR = Path("./cache/papers")
PDF_DOWNLOAD_DELAY_SECONDS = 3.0  # arXiv's requested politeness delay
REQUEST_TIMEOUT_SECONDS = 30


def _download_pdf(pdf_url: str, dest_path: Path) -> None:
    """Download a PDF from arXiv directly via HTTP.

    Needed because arxiv>=4.0.0 dropped Result.download_pdf() --
    the package is metadata-only now and expects callers to fetch
    the file themselves.
    """
    response = requests.get(pdf_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise ValueError(f"Response doesn't look like a PDF (Content-Type: {content_type})")

    dest_path.write_bytes(response.content)


def fetch_papers(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv and download PDFs for the top results.

    Returns a list of paper dicts. Papers that fail to download are
    excluded from the returned list but logged, so the rest of the
    pipeline isn't blocked by one bad paper.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # arxiv.Client handles rate limiting internally (default
    # delay_seconds=3.0 between requests) for the metadata/search
    
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    failed = []

    for result in client.results(search):
        arxiv_id = result.get_short_id()
        pdf_path = CACHE_DIR / f"{arxiv_id}.pdf"

        try:
            if not pdf_path.exists():
                if result.pdf_url is None:
                    raise ValueError("No pdf_url available for this result")

                logger.info(f"Downloading: {result.title[:60]}...")
                _download_pdf(result.pdf_url, pdf_path)
                # Separate politeness delay for the direct arxiv.org
                # PDF download surface, distinct from the API delay
                # the Client already applies to search calls.
                time.sleep(PDF_DOWNLOAD_DELAY_SECONDS)
            else:
                logger.info(f"Cached, skipping download: {result.title[:60]}...")

            # Sanity check: a 0-byte or missing file after "download"
            # means something silently failed -- catch it now rather
            # than letting the parser choke on it later.
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                raise IOError("Downloaded PDF is missing or empty")

            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "pdf_path": str(pdf_path),
                "published": str(result.published.date()),
                "arxiv_id": arxiv_id,
            })

        except Exception as e:
            logger.warning(f"Failed to fetch '{result.title[:60]}...': {e}")
            failed.append({"title": result.title, "arxiv_id": arxiv_id, "error": str(e)})

    if failed:
        logger.warning(f"{len(failed)} paper(s) failed to download and were skipped.")

    return papers, failed


def ingestion_node(state: dict) -> dict:
    """LangGraph node wrapper around fetch_papers."""
    papers, failed = fetch_papers(
        query=state.get("query", "machine learning"),
        max_results=state.get("num_papers", 5),
    )

    state["papers"] = papers
    state.setdefault("errors", [])
    state["errors"].extend([f"Ingestion failed for '{f['title']}': {f['error']}" for f in failed])

    if not papers:
        # Nothing downloaded successfully -- this is a hard stop
        # condition the graph should handle explicitly later (Step on
        # error routing), not something that should silently flow
        # into an empty parse/summarize step.
        state["errors"].append("No papers were successfully ingested.")

    return state