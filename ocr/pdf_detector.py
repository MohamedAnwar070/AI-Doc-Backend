"""Detect whether a PDF is *native* (selectable text) or *scanned* (image-only).

Implements §4.1 of the spec. Used by the worker to skip Google DocAI
billing on digital PDFs.

Decision algorithm
------------------
1. Sample the first ``PROBE_PAGES`` pages.
2. Sum visible characters (excluding whitespace) across the sample.
3. If average characters per page exceeds ``NATIVE_CHARS_PER_PAGE`` AND the
   text is not garbled (mojibake heuristic), classify as ``native``.
4. Otherwise classify as ``scanned``.

Tamil safety
------------
Some Tamil PDFs ship with broken/custom-encoded fonts that produce mojibake
when read as text (``º®¾...`` style). We treat such PDFs as ``scanned`` so
DocAI re-OCRs them with proper Tamil glyph models.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


PdfType = Literal["native", "scanned"]

#: Number of pages to sample for the heuristic.
PROBE_PAGES: int = 3
#: Average characters per page above which the PDF is considered native.
NATIVE_CHARS_PER_PAGE: int = 80
#: Token-level mixed letter/digit ratio above which text is deemed garbled.
_GARBLED_RATIO: float = 0.30
_MIXED_TOKEN = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{2,}")
_TAMIL = re.compile(r"[\u0B80-\u0BFF]")


def _is_garbled(text: str) -> bool:
    tokens = re.findall(r"\S+", text or "")
    if len(tokens) < 10:
        return False
    mixed = sum(1 for t in tokens if _MIXED_TOKEN.search(t))
    return (mixed / max(1, len(tokens))) >= _GARBLED_RATIO


def detect_pdf_type(pdf_path: str | Path) -> PdfType:
    """Return ``'native'`` or ``'scanned'`` for the PDF at ``pdf_path``."""
    path = Path(pdf_path)
    if not path.exists():
        logger.warning("detect_pdf_type: %s not found — assuming scanned", path)
        return "scanned"

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF unavailable — assuming scanned")
        return "scanned"

    try:
        with fitz.open(path) as doc:
            sample_n = min(PROBE_PAGES, len(doc))
            if sample_n == 0:
                return "scanned"
            total_chars = 0
            sample_text = []
            for i in range(sample_n):
                text = doc[i].get_text("text") or ""
                # Count non-whitespace chars
                total_chars += sum(1 for c in text if not c.isspace())
                sample_text.append(text)
            avg = total_chars / sample_n
    except Exception as exc:
        logger.warning("detect_pdf_type failed for %s: %s", path, exc)
        return "scanned"

    joined = "\n".join(sample_text)
    if avg < NATIVE_CHARS_PER_PAGE:
        logger.info(
            "PDF %s: avg %.1f chars/page < %d → scanned",
            path.name,
            avg,
            NATIVE_CHARS_PER_PAGE,
        )
        return "scanned"

    if _is_garbled(joined):
        logger.info(
            "PDF %s: native text appears garbled (custom Tamil font?) → scanned",
            path.name,
        )
        return "scanned"

    logger.info(
        "PDF %s: avg %.1f chars/page → native (tamil_chars=%d)",
        path.name,
        avg,
        len(_TAMIL.findall(joined)),
    )
    return "native"


__all__ = ["PdfType", "detect_pdf_type", "NATIVE_CHARS_PER_PAGE", "PROBE_PAGES"]
