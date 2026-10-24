"""
Layout Detection Engine using PaddleOCR PP-Structure.

Detects document regions (text, table, title, figure, list) and extracts
structured content per region.  Tables are returned as HTML; text regions
as plain strings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageElement:
    """One semantic element detected on a page."""

    type: str                         # "title" | "paragraph" | "table" | "list" | "figure"
    text: str = ""                    # plain text (always present)
    html: Optional[str] = None        # HTML representation (tables only)
    bbox: Optional[List[float]] = None
    language: Optional[str] = None    # detected per-block language
    confidence: float = 0.0
    # Extended fields (Surya / pipeline_v2 compatibility)
    id: Optional[str] = None          # e.g. "p1_b1"
    reading_order: Optional[int] = None
    cleaned: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageResult:
    """Structured result for one PDF page."""

    page_num: int
    elements: List[PageElement] = field(default_factory=list)
    # Optional page-level image metadata (Surya / pipeline_v2 compatibility)
    image: Optional[str] = None       # relative path to rendered image
    width: Optional[int] = None       # image width in pixels
    height: Optional[int] = None      # image height in pixels

    # ----- convenience helpers -----

    def to_plain_text(self) -> str:
        """Flatten to a single plain-text string (backward compatible)."""
        parts: list[str] = []
        for el in self.elements:
            if el.type == "table" and el.html:
                # Produce a readable plain-text version of the HTML table
                parts.append(_html_table_to_text(el.html))
            elif el.text:
                parts.append(el.text)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        d: dict = {
            "page": self.page_num,
            "elements": [e.to_dict() for e in self.elements],
        }
        if self.image is not None:
            d["image"] = self.image
        if self.width is not None:
            d["width"] = self.width
        if self.height is not None:
            d["height"] = self.height
        return d


# ---------------------------------------------------------------------------
# Layout engine
# ---------------------------------------------------------------------------

# Languages supported by PP-Structure's layout model.
# All other languages fall back to 'en' for layout detection.
# (The plain PaddleOCR engine supports more languages, but PP-Structure does not.)
# Do NOT add other langs here — PP-Structure will crash. The hybrid mode in
# paddle_engine.py handles unsupported langs by using 'en' for layout/tables
# and a separate flat PaddleOCR model for the actual text language.
_LAYOUT_SUPPORTED_LANGS = {"en", "ch"}


class LayoutEngine:
    """Wraps PaddleOCR PP-Structure for layout-aware extraction."""

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        recovery: bool = True,
        table_as_html: bool = True,
    ):
        self.use_gpu = use_gpu
        # PP-Structure layout models only support 'en' and 'ch'.
        # For any other language (e.g. 'ta', 'hi') we use 'en' for layout
        # detection while still returning all extracted text as-is.
        if lang not in _LAYOUT_SUPPORTED_LANGS:
            logger.warning(
                f"PP-Structure layout model does not support lang='{lang}'. "
                f"Falling back to 'en' for layout detection. "
                f"OCR text content will still be extracted normally."
            )
            lang = "en"
        self.lang = lang
        self.recovery = recovery
        self.table_as_html = table_as_html
        self._engine = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            from paddleocr import PPStructure

            logger.info(
                f"Initializing PP-Structure (GPU={self.use_gpu}, "
                f"lang={self.lang}, recovery={self.recovery})"
            )
            self._engine = PPStructure(
                show_log=False,
                use_gpu=self.use_gpu,
                lang=self.lang,
                recovery=self.recovery,
                return_ocr_result_in_table=True,
                enable_mkldnn=not self.use_gpu,
            )
            self._initialized = True
            logger.info("PP-Structure initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PP-Structure: {e}")
            raise

    def extract(self, image: Image.Image) -> PageResult:
        """
        Run layout detection + OCR on a single page image.

        Returns a PageResult with structured elements.
        """
        if not self._initialized:
            self.initialize()

        img_array = np.array(image.convert("RGB"))
        raw_results = self._engine(img_array)

        if not raw_results:
            return PageResult(page_num=0, elements=[])

        elements: list[PageElement] = []

        for region in raw_results:
            rtype = region.get("type", "text").lower()
            bbox = region.get("bbox", None)
            res = region.get("res", {})

            if rtype == "table":
                el = self._process_table(res, bbox)
            elif rtype == "title":
                el = self._process_text_region(res, bbox, elem_type="title")
            elif rtype == "figure":
                el = PageElement(type="figure", text="[figure]", bbox=bbox)
            else:
                # text, list, etc.
                mapped = "list" if rtype == "list" else "paragraph"
                el = self._process_text_region(res, bbox, elem_type=mapped)

            if el:
                elements.append(el)

        # Sort by vertical position (top → bottom)
        elements.sort(key=lambda e: (e.bbox[1] if e.bbox else 0))

        return PageResult(page_num=0, elements=elements)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_table(res: dict, bbox) -> PageElement:
        """Extract table as HTML + plain text fallback."""
        html_str = ""
        plain_text = ""

        if isinstance(res, dict):
            html_str = res.get("html", "")
            # Build plain text from the OCR sub-results if available
            ocr_items = res.get("text_region", []) or res.get("cell_ocr_res", [])
            if ocr_items:
                lines = []
                for item in ocr_items:
                    if isinstance(item, dict):
                        lines.append(item.get("text", ""))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        lines.append(str(item[1][0]) if isinstance(item[1], (list, tuple)) else str(item[1]))
                plain_text = "\n".join(l for l in lines if l)

        if not plain_text and html_str:
            plain_text = _html_table_to_text(html_str)

        return PageElement(
            type="table",
            text=plain_text,
            html=html_str or None,
            bbox=bbox,
            confidence=0.0,
        )

    @staticmethod
    def _process_text_region(res, bbox, elem_type: str = "paragraph") -> PageElement:
        """Extract plain text from a text / title region."""
        lines: list[str] = []
        conf_sum = 0.0
        conf_count = 0

        items = res if isinstance(res, list) else []
        for item in items:
            if isinstance(item, dict):
                t = item.get("text", "")
                c = item.get("confidence", 0.0)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                box_part, text_part = item[0], item[1]
                if isinstance(text_part, (list, tuple)):
                    t = str(text_part[0])
                    c = float(text_part[1]) if len(text_part) > 1 else 0.0
                else:
                    t = str(text_part)
                    c = 0.0
            else:
                continue
            if t:
                lines.append(t)
                conf_sum += c
                conf_count += 1

        text = " ".join(lines)
        avg_conf = (conf_sum / conf_count) if conf_count else 0.0

        return PageElement(
            type=elem_type,
            text=text,
            bbox=bbox,
            confidence=round(avg_conf, 4),
        )

    def cleanup(self) -> None:
        if self._engine:
            del self._engine
            self._engine = None
        self._initialized = False
        logger.info("PP-Structure cleaned up")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def _html_table_to_text(html: str) -> str:
    """Best-effort readable text from an HTML <table>."""
    if not html:
        return ""
    # Replace cell boundaries with tab, row boundaries with newline
    text = html
    text = re.sub(r"</td>\s*<td[^>]*>", "\t", text)
    text = re.sub(r"</th>\s*<th[^>]*>", "\t", text)
    text = re.sub(r"</tr>", "\n", text)
    text = _TAG_RE.sub("", text)
    text = text.strip()
    return text
