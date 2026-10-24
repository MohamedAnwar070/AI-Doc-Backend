"""Layout JSON schema — single contract for all OCR engines.

Implements §3 of `.windsurf/subregistrar-ai-spec.md`.

Every extractor (DocAI, Vision AI, pdfplumber, PaddleOCR) MUST be able to
produce a :class:`LayoutDocument` so the frontend coordinate renderer and
the RAG chunker can consume one uniform shape.

Design notes
------------
* Bounding boxes use ABSOLUTE PIXEL coordinates relative to
  (``width_px``, ``height_px``). DocAI / Vision return normalised 0–1
  vertices; the layout builder multiplies by page pixel dimensions.
* ``reading_order`` is a per-page sequential integer assigned by the
  reading-order engine, not by the OCR engine.
* ``section_label`` is extracted by regex from the block text
  (e.g. "Rule 5", "Clause 3(b)", "பிரிவு 12").
* The schema is additive — extra fields are allowed (Pydantic ``extra='allow'``)
  so engines can attach engine-specific metadata without breaking consumers.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


BlockType = Literal[
    "paragraph",
    "heading",
    "title",
    "table",
    "table_cell",
    "list",
    "list_item",
    "figure",
    "header",
    "footer",
]

ExtractionMethod = Literal["native", "ocr", "hybrid"]
LanguageCode = Literal["ta", "en", "mixed", "unknown"]


class TableCell(BaseModel):
    """One cell inside a reconstructed table."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 1


class TableRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    cells: List[TableCell] = Field(default_factory=list)


class LayoutTable(BaseModel):
    """A reconstructed table as a structured grid.

    Engines that already produce HTML (DocAI, Paddle PP-Structure) keep the
    HTML in :class:`LayoutBlock.html` of the corresponding ``"table"`` block;
    this structured form is filled when row/col data is available.
    """

    model_config = ConfigDict(extra="allow")

    table_id: str
    rows: List[TableRow] = Field(default_factory=list)


class LayoutBlock(BaseModel):
    """One semantic block on a page (paragraph, heading, table, etc.).

    All coordinate fields are in pixels, anchored top-left of the page raster
    of dimensions (``width_px``, ``height_px``).
    """

    model_config = ConfigDict(extra="allow")

    block_id: str = Field(..., description="Stable id, e.g. 'p1_b001'")
    type: BlockType = "paragraph"

    # Text
    text: str = ""
    text_tamil: Optional[str] = None
    html: Optional[str] = Field(
        default=None,
        description="HTML representation for tables (legacy DocAI/Paddle path).",
    )

    # Geometry (pixels)
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    # Quality
    confidence: float = 0.0

    # Style (best-effort, populated when engine provides COMPUTE_STYLE_INFO etc.)
    font_size: Optional[float] = None
    is_bold: Optional[bool] = None

    # Semantic
    reading_order: int = 0
    section_label: Optional[str] = Field(
        default=None,
        description="Detected section label, e.g. 'Rule 5' or 'பிரிவு 12'.",
    )
    language: LanguageCode = "unknown"


class LayoutPage(BaseModel):
    """One page worth of structured layout."""

    model_config = ConfigDict(extra="allow")

    page_number: int
    width_px: int = 0
    height_px: int = 0
    dpi: int = 0

    blocks: List[LayoutBlock] = Field(default_factory=list)
    tables: List[LayoutTable] = Field(default_factory=list)


class LayoutDocument(BaseModel):
    """Top-level Layout JSON for one source document.

    This is the response shape of ``GET /api/layout/{job_id}``.
    """

    model_config = ConfigDict(extra="allow")

    document_id: str = Field(..., description="Job id from MongoDB.")
    source_file: str = ""
    extraction_method: ExtractionMethod = "ocr"
    total_pages: int = 0
    pages: List[LayoutPage] = Field(default_factory=list)


__all__ = [
    "BlockType",
    "ExtractionMethod",
    "LanguageCode",
    "LayoutBlock",
    "LayoutDocument",
    "LayoutPage",
    "LayoutTable",
    "TableCell",
    "TableRow",
]
