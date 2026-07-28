"""
Ingestion Agent (Agent 1).

Searches arXiv for papers matching a query and downloads their PDFs
to a local cache, handling per-paper failures gracefully so one bad
download does not block the rest of the batch.

Key implementation notes
-------------------------
- Uses `arxiv.Client().results(search)` instead of the deprecated
  `search.results()` method (removed in arxiv>=2.x).
- `arxiv>=4.0.0` removed `Result.download_pdf()`; the library is now
  metadata/search-only. PDF bytes are fetched manually via `requests`
  using `result.pdf_url`. Confirmed against the installed version
  (4.0.0) directly rather than assumed from documentation -- re-verify
  against a real run if this dependency is ever upgraded again.
- Rate limits are respected on two separate surfaces: the arXiv API
  (handled internally by `arxiv.Client`'s `delay_seconds`) and direct
  PDF downloads from arxiv.org (handled manually via `time.sleep`),
  since these are distinct endpoints with independent rate limits.
- Each paper's download is wrapped in a try/except so a single failure
  (network error, malformed file, 404) is recorded and skipped rather
  than aborting the whole batch.
- PDFs already present in the cache are not re-downloaded.
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
    """
    Download a PDF from arXiv via direct HTTP request.

    Required because `arxiv>=4.0.0` no longer provides
    `Result.download_pdf()`; the package is metadata-only and expects
    callers to fetch the file themselves.

    Parameters
    ----------
    pdf_url : str
        Direct URL to the PDF, as provided by an arXiv search result.
    dest_path : Path
        Local filesystem path to write the downloaded PDF to.

    Raises
    ------
    requests.exceptions.RequestException
        If the HTTP request fails (for example, connection or timeout
        errors) or returns an error status.
    ValueError
        If the response does not appear to be a valid PDF.
    """
    response = requests.get(pdf_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise ValueError(f"Response doesn't look like a PDF (Content-Type: {content_type})")

    dest_path.write_bytes(response.content)


def fetch_papers(query: str, max_results: int = 5) -> tuple[list[dict], list[dict]]:
    """
    Search arXiv and download PDFs for the top matching results.

    Parameters
    ----------
    query : str
        arXiv search query string.
    max_results : int, optional
        Maximum number of results to fetch, by default 5.

    Returns
    -------
    tuple[list[dict], list[dict]]
        A tuple of `(papers, failed)`:
        - `papers`: successfully ingested papers, each represented as
          a dict with title, authors, abstract, pdf_path, published
          date, and arxiv_id.
        - `failed`: papers that could not be downloaded, each
          represented as a dict with title, arxiv_id, and error.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # arxiv.Client handles rate limiting internally (default
    # delay_seconds=3.0 between requests) for the metadata/search
    # calls, which the original plan mentioned but never actually
    # implemented.
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
    """
    LangGraph node wrapper around `fetch_papers`.

    Parameters
    ----------
    state : dict
        Current graph state. Reads `query` and `num_papers`.

    Returns
    -------
    dict
        Partial state update containing only this node's new
        contributions: `papers` and `errors`.

    Important
    ---------
    Must return a partial dict, not the full mutated state. `errors`
    is an `Annotated[list, operator.add]` reducer field in
    `ResearchState`; returning the full accumulated list (rather than
    only this node's new items) causes LangGraph to double/triple-count
    entries. Keep this consistent with the reducer definitions in
    `graph/state.py` and `graph/workflow.py`.
    """
    papers, failed = fetch_papers(
        query=state.get("query", "machine learning"),
        max_results=state.get("num_papers", 5),
    )

    new_errors = [f"Ingestion failed for '{f['title']}': {f['error']}" for f in failed]
    if not papers:
        new_errors.append("No papers were successfully ingested.")

    if failed:
        logger.warning(f"{len(failed)} paper(s) failed to download and were skipped.")

    return {
        "papers": papers,
        "errors": new_errors,
    }