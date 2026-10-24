"""Word document worker.

Extracts text and layout-like structure from ``.docx`` files using
``python-docx`` and stores it through the same job output pipeline used by
PDF OCR and Excel extraction.
"""
from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Iterable

from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from core.source_preview import extract_pdf_text_pages

logger = logging.getLogger(__name__)


def _iter_block_items(document: DocxDocument) -> Iterable[Paragraph | Table]:
    """Yield top-level paragraphs and tables in document order."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _safe_text(value: str | None) -> str:
    return (value or "").replace("\t", "    ")


def _escape_text(value: str | None) -> str:
    return html.escape(_safe_text(value), quote=False).replace("\n", "<br>")


def _paragraph_text(paragraph: Paragraph) -> str:
    return _safe_text(paragraph.text).strip()


def _paragraph_html(paragraph: Paragraph, text: str) -> str:
    style_name = (paragraph.style.name if paragraph.style is not None else "").lower()
    tag = "p"
    css_class = "docx-paragraph"

    if "title" == style_name:
        tag = "h1"
        css_class = "docx-title"
    elif style_name.startswith("heading"):
        level = "".join(ch for ch in style_name if ch.isdigit())
        tag = f"h{min(max(int(level or '2'), 1), 4)}"
        css_class = "docx-heading"
    elif "list" in style_name:
        css_class = "docx-list"

    parts: list[str] = []
    for run in paragraph.runs:
        run_text = _escape_text(run.text)
        if not run_text:
            continue
        if run.bold:
            run_text = f"<strong>{run_text}</strong>"
        if run.italic:
            run_text = f"<em>{run_text}</em>"
        if run.underline:
            run_text = f"<u>{run_text}</u>"
        parts.append(run_text)

    inner = "".join(parts) or _escape_text(text)
    return f'<{tag} class="{css_class}">{inner}</{tag}>'


def _paragraph_type(paragraph: Paragraph) -> str:
    style_name = (paragraph.style.name if paragraph.style is not None else "").lower()
    if style_name == "title":
        return "title"
    if style_name.startswith("heading"):
        return "heading"
    if "list" in style_name:
        return "list"
    return "paragraph"


def _table_cell_text(cell) -> str:
    return "\n".join(
        _paragraph_text(paragraph)
        for paragraph in cell.paragraphs
        if _paragraph_text(paragraph)
    ).strip()


def _table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [_table_cell_text(cell) for cell in row.cells]
        if any(cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _table_to_html(table: Table) -> str:
    rows_html: list[str] = ['<table class="docx-table">']
    for row_idx, row in enumerate(table.rows):
        rows_html.append("<tr>")
        tag = "th" if row_idx == 0 else "td"
        for cell in row.cells:
            text = _escape_text(_table_cell_text(cell))
            rows_html.append(f"<{tag}>{text}</{tag}>")
        rows_html.append("</tr>")
    rows_html.append("</table>")
    return "".join(rows_html)


class WordWorker:
    """Process one DOCX file end-to-end: extract, store, and index."""

    def __init__(
        self,
        job_id: str,
        docx_path: Path,
        storage,
        progress_tracker,
        resume_handler,
        job_manager=None,
    ):
        self.job_id = job_id
        self.docx_path = docx_path
        self.storage = storage
        self.progress_tracker = progress_tracker
        self.resume_handler = resume_handler
        self.job_manager = job_manager

    def process(self) -> None:
        logger.info("WordWorker: job=%s file=%s", self.job_id, self.docx_path.name)
        try:
            self.progress_tracker.start_job(self.job_id)
            full_text, elements = self._extract_docx()
            preview_pages = self._extract_source_preview_pages()
            self._store(full_text, elements, preview_pages)
            self._build_outputs()

            if self.job_manager is not None and self.job_manager.loop is not None:
                self.job_manager.schedule_indexing(
                    self.job_id,
                    filename=self.docx_path.name,
                    force=True,
                )
                logger.info("WordWorker: scheduled indexing for job %s", self.job_id)
        except Exception as exc:
            msg = f"Word processing failed: {exc}"
            logger.error(msg, exc_info=True)
            self.progress_tracker.mark_failed(self.job_id, msg)
            raise

    def _extract_docx(self) -> tuple[str, list[dict]]:
        from docx import Document

        document = Document(str(self.docx_path))
        text_blocks: list[str] = []
        elements: list[dict] = []
        order = 0

        for block in _iter_block_items(document):
            if isinstance(block, Paragraph):
                text = _paragraph_text(block)
                if not text:
                    continue
                element_type = _paragraph_type(block)
                text_blocks.append(text)
                elements.append({
                    "id": f"docx-{order}",
                    "type": element_type,
                    "text": text,
                    "html": _paragraph_html(block, text),
                    "reading_order": order,
                    "confidence": 1.0,
                    "cleaned": True,
                })
            else:
                text = _table_to_text(block)
                if not text:
                    continue
                text_blocks.append(text)
                elements.append({
                    "id": f"docx-{order}",
                    "type": "table",
                    "text": text,
                    "html": _table_to_html(block),
                    "reading_order": order,
                    "confidence": 1.0,
                    "cleaned": True,
                })
            order += 1

        full_text = "\n\n".join(text_blocks).strip()
        if not full_text:
            raise ValueError("No readable text found in DOCX")

        logger.info(
            "WordWorker: extracted %d chars and %d structural elements from %s",
            len(full_text),
            len(elements),
            self.docx_path.name,
        )
        return full_text, elements

    def _extract_source_preview_pages(self) -> list[dict]:
        if not hasattr(self.storage, "get_source_pdf_path"):
            return []
        source_pdf = self.storage.get_source_pdf_path(self.job_id)
        if not source_pdf.exists():
            return []
        pages = extract_pdf_text_pages(source_pdf)
        if not any(page.get("elements") for page in pages):
            return []
        return pages

    def _store(self, text: str, elements: list[dict], preview_pages: list[dict] | None = None) -> None:
        if preview_pages:
            text_parts: list[str] = []
            for page in preview_pages:
                page_num = int(page.get("page") or 1)
                page_text = "\n\n".join(
                    element.get("text", "")
                    for element in page.get("elements", [])
                    if element.get("text")
                ).strip()
                if page_text:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                self.storage.save_page_structure(self.job_id, page_num, page)
                self.progress_tracker.update_progress(self.job_id, page_num, success=True)

            result_path = self.storage.get_result_file_path(self.job_id)
            result_path.write_text("\n\n".join(text_parts).strip() or text, encoding="utf-8")
            logger.info(
                "WordWorker: stored page-preserved preview extraction for %s (%d pages)",
                self.job_id,
                len(preview_pages),
            )
            return

        self.storage.append_text(self.job_id, text, page_num=1)
        page_dict = {
            "page": 1,
            "width": 900,
            "height": 1200,
            "elements": elements,
        }
        self.storage.save_page_structure(self.job_id, 1, page_dict)
        self.progress_tracker.update_progress(self.job_id, 1, success=True)

        total_pages = int(self.progress_tracker.get_status(self.job_id).get("total_pages", 1) or 1)
        for page_num in range(2, total_pages + 1):
            self.storage.save_page_structure(self.job_id, page_num, {
                "page": page_num,
                "width": 900,
                "height": 1200,
                "elements": [],
            })
            self.progress_tracker.update_progress(self.job_id, page_num, success=True)
        logger.info("WordWorker: stored text and structure for job %s", self.job_id)

    def _build_outputs(self) -> None:
        if hasattr(self.storage, "build_combined_structured_result"):
            self.storage.build_combined_structured_result(self.job_id)
        if hasattr(self.storage, "build_html_result"):
            self.storage.build_html_result(self.job_id)
        logger.info("WordWorker: built outputs for job %s", self.job_id)
