"""Build :class:`LayoutDocument` from per-page OCR JSONs.

The OCR worker already writes ``pages/page_NNNN.json`` for every job using
``PageResult.to_dict()``::

    {"page": 1, "elements": [{"type": "...", "text": "...",
                              "bbox": [x0,y0,x1,y1], "confidence": 0.97,
                              "html": null, "language": null}, ...]}

This module aggregates those into the spec §3 :class:`LayoutDocument`:
* normalised DocAI/Vision bboxes (0..1) → absolute pixel coords using the
  page raster size derived from the source PDF.
* bilingual section-label detection
* reading-order assignment
* heading classification kept (already done by engines)

The builder is filesystem-driven so it works for **any** OCR engine
(DocAI, Vision AI, Paddle, Remote) without coupling.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from .layout_schema import (
    BlockType,
    ExtractionMethod,
    LanguageCode,
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
    LayoutTable,
)
from .reading_order import assign_reading_order

logger = logging.getLogger(__name__)


# ── Defaults for engines that don't produce page geometry ──────────────────

#: Render DPI assumed when the worker rasterised at ~300 DPI for OCR.
DEFAULT_DPI: int = 300
#: A4 at 300 DPI fallback so a missing source PDF doesn't break the renderer.
DEFAULT_PAGE_WIDTH_PX: int = 2480
DEFAULT_PAGE_HEIGHT_PX: int = 3508


# ── BlockType normalisation ────────────────────────────────────────────────

_BLOCK_TYPE_MAP: dict[str, BlockType] = {
    "title": "title",
    "heading": "heading",
    "paragraph": "paragraph",
    "text": "paragraph",
    "table": "table",
    "list": "list",
    "list_item": "list_item",
    "figure": "figure",
    "header": "header",
    "footer": "footer",
}


def _coerce_block_type(raw: Any) -> BlockType:
    return _BLOCK_TYPE_MAP.get(str(raw or "").lower(), "paragraph")


def _coerce_language(raw: Any) -> LanguageCode:
    code = str(raw or "").lower().strip()
    if code.startswith("ta"):
        return "ta"
    if code.startswith("en"):
        return "en"
    if code in {"mixed", "bilingual"}:
        return "mixed"
    return "unknown"


# ── BBox conversion (normalised 0..1 vs pixel) ─────────────────────────────

def _is_normalised(bbox: list[float]) -> bool:
    """Heuristic: DocAI / Vision return all coords in [0, 1]."""
    return all(0.0 <= c <= 1.5 for c in bbox)


def _bbox_to_pixels(
    bbox: Optional[list[float]],
    width_px: int,
    height_px: int,
) -> tuple[float, float, float, float]:
    """Return (x, y, w, h) in absolute pixel coordinates."""
    if not bbox or len(bbox) != 4:
        return 0.0, 0.0, 0.0, 0.0
    x0, y0, x1, y1 = (float(c) for c in bbox)
    if _is_normalised([x0, y0, x1, y1]):
        x0 *= width_px
        x1 *= width_px
        y0 *= height_px
        y1 *= height_px
    x = min(x0, x1)
    y = min(y0, y1)
    w = max(0.0, abs(x1 - x0))
    h = max(0.0, abs(y1 - y0))
    return x, y, w, h


# ── Page geometry ──────────────────────────────────────────────────────────

def _read_page_geometry(pdf_path: Optional[Path]) -> dict[int, tuple[int, int, int]]:
    """Map ``page_number → (width_px, height_px, dpi)`` from a source PDF.

    Falls back to A4 @ 300 DPI when the file is missing or unreadable.
    """
    geometry: dict[int, tuple[int, int, int]] = {}
    if not pdf_path or not pdf_path.exists():
        return geometry
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available — using default page geometry")
        return geometry
    try:
        with fitz.open(pdf_path) as doc:
            for idx, page in enumerate(doc, start=1):
                rect = page.rect
                width_pt, height_pt = float(rect.width), float(rect.height)
                # PDF points → pixels at DEFAULT_DPI (1 pt = 1/72 in)
                width_px = int(round(width_pt * DEFAULT_DPI / 72.0))
                height_px = int(round(height_pt * DEFAULT_DPI / 72.0))
                geometry[idx] = (width_px, height_px, DEFAULT_DPI)
    except Exception as exc:
        logger.warning("Could not read PDF geometry from %s: %s", pdf_path, exc)
    return geometry


# ── Element → LayoutBlock ──────────────────────────────────────────────────

def _build_block(
    *,
    page_number: int,
    block_index: int,
    element: dict[str, Any],
    width_px: int,
    height_px: int,
) -> LayoutBlock:
    x, y, w, h = _bbox_to_pixels(element.get("bbox"), width_px, height_px)
    return LayoutBlock(
        block_id=f"p{page_number}_b{block_index:03d}",
        type=_coerce_block_type(element.get("type")),
        text=str(element.get("text") or "").strip(),
        html=element.get("html") or None,
        x=x,
        y=y,
        width=w,
        height=h,
        confidence=float(element.get("confidence") or 0.0),
        language=_coerce_language(element.get("language")),
        # reading_order + section_label assigned later by reading_order engine
    )


# ── Public API ─────────────────────────────────────────────────────────────

def build_from_page_jsons(
    *,
    job_id: str,
    source_file: str,
    page_jsons: List[dict[str, Any]],
    pdf_path: Optional[Path] = None,
    extraction_method: ExtractionMethod = "ocr",
) -> LayoutDocument:
    """Aggregate a list of per-page OCR dicts into a :class:`LayoutDocument`.

    Parameters
    ----------
    job_id:
        MongoDB job id (becomes ``document_id``).
    source_file:
        Original filename (display only).
    page_jsons:
        List of dicts as produced by ``PageResult.to_dict()``.
    pdf_path:
        Optional source PDF on disk — used to read true page dimensions.
    extraction_method:
        ``"native"`` for pdfplumber-only path, ``"ocr"`` otherwise.
    """
    geometry = _read_page_geometry(pdf_path)

    pages: List[LayoutPage] = []
    for page_data in page_jsons:
        # VLM engines already save LayoutPage.model_dump(). Do not route that
        # data through the legacy {page, elements, bbox} converter: doing so
        # drops its table objects, page raster dimensions and block geometry.
        if isinstance(page_data.get("blocks"), list):
            page_number = int(page_data.get("page_number") or page_data.get("page") or len(pages) + 1)
            fallback_width, fallback_height, fallback_dpi = geometry.get(
                page_number,
                (DEFAULT_PAGE_WIDTH_PX, DEFAULT_PAGE_HEIGHT_PX, DEFAULT_DPI),
            )
            width_px = int(page_data.get("width_px") or fallback_width)
            height_px = int(page_data.get("height_px") or fallback_height)
            dpi = int(page_data.get("dpi") or fallback_dpi)
            blocks: List[LayoutBlock] = []
            for raw_block in page_data["blocks"]:
                if not isinstance(raw_block, dict):
                    continue
                try:
                    blocks.append(LayoutBlock.model_validate(raw_block))
                except Exception as exc:
                    logger.warning("Skipping invalid structured block on page %d: %s", page_number, exc)
            tables: List[LayoutTable] = []
            for raw_table in page_data.get("tables") or []:
                if not isinstance(raw_table, dict):
                    continue
                try:
                    tables.append(LayoutTable.model_validate(raw_table))
                except Exception as exc:
                    logger.warning("Skipping invalid structured table on page %d: %s", page_number, exc)
            pages.append(LayoutPage(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                dpi=dpi,
                blocks=blocks,
                tables=tables,
                image=page_data.get("image"),
            ))
            continue

        page_number = int(page_data.get("page") or len(pages) + 1)
        width_px, height_px, dpi = geometry.get(
            page_number,
            (DEFAULT_PAGE_WIDTH_PX, DEFAULT_PAGE_HEIGHT_PX, DEFAULT_DPI),
        )

        blocks: List[LayoutBlock] = []
        for idx, element in enumerate(page_data.get("elements") or []):
            if not isinstance(element, dict):
                continue
            block = _build_block(
                page_number=page_number,
                block_index=idx + 1,
                element=element,
                width_px=width_px,
                height_px=height_px,
            )
            if not block.text and not block.html:
                continue
            blocks.append(block)

        # Reading order + section labels (mutates blocks)
        blocks = assign_reading_order(blocks)

        pages.append(
            LayoutPage(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                dpi=dpi,
                blocks=blocks,
                tables=[],  # populated only when an engine emits structured tables
            )
        )

    pages.sort(key=lambda p: p.page_number)

    return LayoutDocument(
        document_id=job_id,
        source_file=source_file,
        extraction_method=extraction_method,
        total_pages=len(pages),
        pages=pages,
    )


def build_from_disk(
    *,
    job_id: str,
    source_file: str,
    pages_dir: Path,
    pdf_path: Optional[Path] = None,
    extraction_method: ExtractionMethod = "ocr",
) -> LayoutDocument:
    """Read ``pages/page_NNNN.json`` files from disk and build the document."""
    page_jsons: List[dict[str, Any]] = []
    if pages_dir.exists():
        for pf in sorted(pages_dir.glob("page_*.json")):
            try:
                page_jsons.append(json.loads(pf.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Could not parse %s: %s", pf, exc)
    return build_from_page_jsons(
        job_id=job_id,
        source_file=source_file,
        page_jsons=page_jsons,
        pdf_path=pdf_path,
        extraction_method=extraction_method,
    )


__all__ = ["build_from_disk", "build_from_page_jsons"]
