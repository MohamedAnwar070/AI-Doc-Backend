"""Pure helpers for :mod:`api.upload`.

Kept in a standalone module so unit tests can import them without
pulling the heavyweight OCR worker stack (PyMuPDF, PaddleOCR, etc.).
Anything that needs FastAPI, MongoDB, or a model lives in
``api/upload.py``; anything that's just string/path manipulation
lives here.
"""
from __future__ import annotations

import os
import re

#: Anything outside this set is replaced with ``_`` in the saved filename.
_SAFE_FILENAME_RE: re.Pattern[str] = re.compile(r"[^A-Za-z0-9._-]+")

#: Magic bytes that identify a PDF file.
PDF_MAGIC: bytes = b"%PDF-"

#: Magic bytes that identify a ZIP-based file (xlsx/docx are zip internally).
XLSX_MAGIC: bytes = b"PK\x03\x04"

#: Accepted upload language hints (``auto`` triggers auto-detection).
KNOWN_LANGS: frozenset[str] = frozenset({
    "auto", "en", "ta", "hi", "ml", "te", "kn",
})

#: Accepted file extensions.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".xlsx", ".docx"})

#: 8 KiB chunk size for streaming the upload to disk.
COPY_CHUNK_BYTES: int = 8 * 1024

#: Hard cap on the saved filename stem so we don't break filesystem limits.
_MAX_STEM_CHARS: int = 120


def sanitise_filename(raw: str | None, ext_override: str | None = None) -> str:
    """Return a filesystem-safe, traversal-proof filename.

    Preserves ``.pdf``, ``.xlsx`` or ``.docx`` suffix.  Empty / dot-only inputs map
    to ``"upload.pdf"``.

    Examples
    --------
    >>> sanitise_filename("../../etc/passwd")
    'passwd.pdf'
    >>> sanitise_filename("Stamp Act-1.xlsx")
    'Stamp_Act-1.xlsx'
    >>> sanitise_filename("Notice Draft.docx")
    'Notice_Draft.docx'
    """
    base = os.path.basename(raw or "").strip()
    if not base or base in {".", ".."}:
        return f"upload{ext_override or '.pdf'}"

    stem, ext = os.path.splitext(base)
    ext_lower = ext.lower()

    if ext_lower not in ALLOWED_EXTENSIONS:
        stem = base
        ext_lower = ext_override or ".pdf"

    cleaned = _SAFE_FILENAME_RE.sub("_", stem).strip("._-") or "upload"
    return f"{cleaned[:_MAX_STEM_CHARS]}{ext_lower}"


def validate_language(lang: str | None) -> str:
    """Return ``lang`` if it is in :data:`KNOWN_LANGS`, else ``'auto'``."""
    return lang if (lang in KNOWN_LANGS) else "auto"
