"""
PDF generator for the Agentic Research Paper Summarizer.

Converts the generated Markdown literature review into a
professionally formatted PDF using ReportLab.

Works on Windows, Linux and macOS.
"""

from io import BytesIO
from datetime import datetime
import re

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
)
from reportlab.pdfbase import pdfmetrics


def _clean_inline_markdown(text: str) -> str:
    """Remove simple Markdown formatting."""

    # **bold**
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # *italic*
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)

    # `code`
    text = re.sub(r"`(.*?)`", r"<font face='Courier'>\1</font>", text)

    return text


def markdown_to_pdf(
    markdown_text: str,
    query: str,
    num_papers: int,
    model: str,
    mode: str,
) -> bytes:
    """
    Convert Markdown literature review into a PDF.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=(8.27 * inch, 11.69 * inch),  # A4
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    story = []

    # ------------------------------------------------------------------
    # Cover
    # ------------------------------------------------------------------

    story.append(Paragraph("Research Literature Review", title_style))
    story.append(Spacer(1, 0.35 * inch))

    story.append(Paragraph(f"<b>Research Query:</b> {query}", body))
    story.append(
        Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%d %B %Y %H:%M')}",
            body,
        )
    )
    story.append(Paragraph(f"<b>Number of Papers:</b> {num_papers}", body))
    story.append(Paragraph(f"<b>Processing Mode:</b> {mode}", body))
    story.append(Paragraph(f"<b>Model:</b> {model}", body))

    story.append(Spacer(1, 0.4 * inch))

    # ------------------------------------------------------------------
    # Markdown parsing
    # ------------------------------------------------------------------

    lines = markdown_text.splitlines()

    bullets = []

    def flush_bullets():
        nonlocal bullets

        if bullets:
            story.append(
                ListFlowable(
                    [
                        ListItem(Paragraph(_clean_inline_markdown(x), body))
                        for x in bullets
                    ],
                    bulletType="bullet",
                )
            )
            bullets = []

    for raw in lines:

        line = raw.strip()

        if not line:
            flush_bullets()
            story.append(Spacer(1, 0.15 * inch))
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(
                Paragraph(
                    _clean_inline_markdown(line[2:]),
                    h1,
                )
            )
            continue

        if line.startswith("## "):
            flush_bullets()
            story.append(
                Paragraph(
                    _clean_inline_markdown(line[3:]),
                    h2,
                )
            )
            continue

        if line.startswith("### "):
            flush_bullets()
            story.append(
                Paragraph(
                    _clean_inline_markdown(line[4:]),
                    h3,
                )
            )
            continue

        if line.startswith("- "):
            bullets.append(line[2:])
            continue

        if line.startswith("* "):
            bullets.append(line[2:])
            continue

        if line.startswith("---"):
            flush_bullets()
            story.append(Spacer(1, 0.2 * inch))
            continue

        flush_bullets()
        story.append(
            Paragraph(
                _clean_inline_markdown(line),
                body,
            )
        )

    flush_bullets()

    # ------------------------------------------------------------------

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(
            7.7 * inch,
            0.4 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    pdf = buffer.getvalue()
    buffer.close()

    return pdf