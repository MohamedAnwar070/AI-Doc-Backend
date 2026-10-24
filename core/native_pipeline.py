"""Native PDF pipeline runner.

Glue between :mod:`ocr.pdf_detector`, :mod:`ocr.native_extractor` and the
existing :class:`storage.file_handler.FileHandler`. When a PDF has a real
text layer, this runner produces the same on-disk artefacts (per-page JSONs,
``result.txt``, ``result.json``) that the OCR worker produces — without
calling Google DocAI / Vision.

Usage
-----
::

    from core.native_pipeline import try_run_native_pipeline

    used_native = try_run_native_pipeline(
        job_id=job_id,
        pdf_path=pdf_path,
        storage=file_handler,
        progress_tracker=tracker,
    )
    if not used_native:
        # fall through to the OCR worker path
        ...

Behaviour
---------
* Returns ``False`` (no side-effects) when:
  - ``settings.use_native_extractor`` is ``False``
  - PDF is classified ``scanned`` by :func:`ocr.pdf_detector.detect_pdf_type`
  - pdfplumber is not installed
* Returns ``True`` after writing ``pages/page_NNNN.json`` for every page
  and the flat ``result.txt``. The combined ``result.json`` is built using
  the storage handler's existing helper.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from config import settings
from ocr.native_extractor import NativeExtractionError, extract_native
from ocr.pdf_detector import detect_pdf_type

logger = logging.getLogger(__name__)


def _flatten_page_text(page_dict: dict[str, Any]) -> str:
    """Reproduce the flattening rule used by :class:`PageResult.to_plain_text`."""
    parts: list[str] = []
    for el in page_dict.get("elements", []) or []:
        text = (el.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def try_run_native_pipeline(
    *,
    job_id: str,
    pdf_path: str | Path,
    storage: Any,
    progress_tracker: Optional[Any] = None,
) -> bool:
    """Attempt the native (non-OCR) pipeline. Return ``True`` on success.

    Parameters
    ----------
    job_id:
        Job identifier (used for storage paths).
    pdf_path:
        Path to the source PDF on disk.
    storage:
        Anything implementing :meth:`save_page_structure`,
        :meth:`append_text`, :meth:`build_combined_structured_result`. The
        repository's :class:`storage.file_handler.FileHandler` qualifies.
    progress_tracker:
        Optional. Anything with ``update_progress(job_id, page_num, success)``.
    """
    if not getattr(settings, "use_native_extractor", False):
        return False

    pdf_path = Path(pdf_path)
    pdf_type = detect_pdf_type(pdf_path)
    if pdf_type != "native":
        logger.info(
            "native_pipeline: %s detected as %s — skipping",
            pdf_path.name,
            pdf_type,
        )
        return False

    try:
        page_dicts = extract_native(pdf_path)
    except NativeExtractionError as exc:
        logger.warning("native_pipeline: extraction failed: %s — falling back", exc)
        return False

    if not page_dicts:
        logger.info("native_pipeline: no pages extracted — falling back")
        return False

    for page_dict in page_dicts:
        page_num = int(page_dict.get("page", 0))
        if page_num <= 0:
            continue
        try:
            storage.save_page_structure(job_id, page_num, page_dict)
            storage.append_text(job_id, _flatten_page_text(page_dict), page_num)
        except Exception as exc:
            logger.exception(
                "native_pipeline: failed to persist page %d: %s", page_num, exc
            )
            return False
        if progress_tracker is not None:
            try:
                progress_tracker.update_progress(job_id, page_num, True)
            except Exception:  # progress is best-effort
                pass

    # Build the combined result.json so downstream consumers (result API,
    # search indexer) see a coherent document.
    if hasattr(storage, "build_combined_structured_result"):
        try:
            storage.build_combined_structured_result(job_id)
        except Exception as exc:
            logger.warning("native_pipeline: combined result build failed: %s", exc)

    logger.info(
        "native_pipeline: job %s extracted %d pages via pdfplumber (no OCR)",
        job_id,
        len(page_dicts),
    )
    return True


__all__ = ["try_run_native_pipeline"]
