"""PDF OCR worker.

Consumes uploaded PDFs from a job's filesystem location and produces
OCR text + structured page results, both written through the
:class:`storage.file_handler.FileHandler` abstraction.

Pipeline per page::

    PDF page
        → native text fast path  (PyMuPDF)         ← when text layer exists
        → DocAI path: single-page PDF bytes        ← when engine=google_docai
          (DocAI renders internally at optimal DPI — no client PNG needed)
        → PNG render at language-aware DPI         ← paddle / remote fallback
        → engine = OCRFactory.get_engine(...)
            ├── GoogleDocAIEngine: PDF bytes → extract_from_pdf()
            ├── PaddleVLEngine: VLM inference → extract_page()   ← NEW
            ├── PaddleEngine: structured / flat
            └── RemoteOCREngine: HTTP client
        → core.text_normalizer.normalize(...)      ← Tamil NFC, confusables, ws
        → write to storage

Concurrency: the worker uses ``concurrent.futures.ProcessPoolExecutor``
so each page runs in its own process. This is mandatory for PaddleOCR
which is *not* threadsafe and which holds large native buffers.

DocAI PDF-bytes strategy
-------------------------
For the ``google_docai`` engine the worker extracts a single-page PDF
slice (PyMuPDF ``Document.select([page_idx])`` → in-memory bytes) and
passes it as the payload instead of a PNG raster image.  DocAI renders
the PDF internally at the highest possible DPI using its proprietary
pipeline — delivering better character accuracy than any client-side
Pillow render, especially for Tamil glyphs at small font sizes.
"""
from __future__ import annotations

import io
import logging
import multiprocessing
import re
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from pathlib import Path
from typing import Any, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from core.llm_cleaner import is_llm_clean_enabled, llm_clean_elements, llm_clean_text
from core.progress_tracker import ProgressTracker
from core.text_normalizer import is_predominantly_tamil, normalize
from ocr.factory import OCRFactory
from storage.resume_handler import ResumeHandler

logger = logging.getLogger(__name__)


# ── Tunables (configurable knobs are in :mod:`config`) ─────────────────────

#: Min characters of native text to consider the page text-layer present.
NATIVE_TEXT_THRESHOLD: int = 50

#: Strong per-page native-text signal for doc-type auto-detection.
_NATIVE_TEXT_STRONG_CHARS: int = 200

#: Strong per-page native-text signal based on word count.
_NATIVE_TEXT_STRONG_WORDS: int = 40

#: Per-page OCR timeout in seconds.
PAGE_TIMEOUT_SECONDS: int = 300

#: Tamil glyph Unicode range, used by the native-text Tamil detector.
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

#: Number of pages we sample for language auto-detection.
_LANG_PROBE_PAGE_COUNT: int = 3

#: Number of Tamil glyphs needed before we conclude a doc is Tamil.
_TAMIL_DETECT_THRESHOLD: int = 5

#: Minimum word count before garbled-text heuristic fires.
_GARBLED_MIN_WORDS: int = 10

#: If more than this fraction of word-tokens are letter–digit mixed, text is
#: considered garbled (e.g. Adobe Scan Tamil: "6T600r", "bn6rfl6or").
_GARBLED_MIXED_RATIO: float = 0.25

_MIXED_TOKEN_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{2,}")

# ── Scanned / native PDF auto-detection constants ──────────────────────────

#: Number of pages to sample for doc-type auto-detection.
_DOC_TYPE_PROBE_PAGES: int = 3

#: Average bytes per page above which a PDF is strongly suspected scanned.
_SCANNED_SIZE_THRESHOLD_BYTES_PER_PAGE: int = 80_000  # ~80 KB/page

#: If a probe page contains images, add this weight toward "scanned".
_IMAGE_WEIGHT: int = 3

#: If a probe page has low text-to-page-area ratio, add this weight.
_LOW_TEXT_DENSITY_WEIGHT: int = 2

#: If probe text is garbled, add this weight toward "scanned".
_GARBLED_TEXT_WEIGHT: int = 3

#: If filename contains scan-related keywords, add this weight.
_FILENAME_HINT_WEIGHT: int = 2

#: Threshold score; >= this means "scanned", < means "native".
_SCANNED_SCORE_THRESHOLD: int = 3

_SCANNED_FILENAME_RE = re.compile(
    r"(scan|scanner|adobe scan|camscanner|office lens|"
    r"scanned|document scan|photo|image|cam|phone|mobile)",
    re.IGNORECASE,
)


def _is_garbled_native_text(text: str) -> bool:
    """Return True if *text* is garbled and should NOT be used as native text.

    Handles two classes of bad text layers:

    Case 1 — Legacy Tamil font encoding (Bamini, TAB, TAM, TSCII, Vijaya…)
        These fonts map Tamil glyphs into the Latin-1 Supplement block (U+0080
        to U+00FF, i.e. byte values 128-255). PyMuPDF faithfully extracts the
        bytes and produces text that looks like ``ªêòô¢ð´î¢î «õí¢´ñ¢``.
        Real Unicode Tamil lives in U+0B80-U+0BFF (all > 255).  So if more
        than 5 % of the characters sit in the 128-255 band we know the PDF
        uses a legacy font and the text layer is unusable — send to OCR.

    Case 2 — Adobe Scan / CamScanner alphanumeric noise
        Poor-quality embedded OCR text for Tamil scanned images contains
        tokens that mix letters and digits (``6T600r``, ``bn6rfl6or``).
        We detect this by the fraction of such mixed tokens.
    """
    if not text:
        return False

    # ── Case 1: Legacy font mojibake (Extended ASCII / Latin-1 Supplement) ──
    # Real Unicode Tamil is > U+00FF. Clean English / numbers are <= U+007F.
    # Characters in [128, 255] should be essentially absent in well-formed
    # Tamil+English PDFs using Unicode fonts.
    text_len = len(text)
    if text_len > 30:
        extended_ascii_count = sum(1 for c in text if 128 <= ord(c) <= 255)
        if (extended_ascii_count / text_len) > 0.05:
            return True  # legacy font mojibake

    # ── Case 2: Adobe Scan alphanumeric noise ─────────────────────────────
    words = re.findall(r"\b\w+\b", text)
    if len(words) < _GARBLED_MIN_WORDS:
        return False
    mixed_count = sum(1 for w in words if _MIXED_TOKEN_RE.fullmatch(w))
    return (mixed_count / len(words)) > _GARBLED_MIXED_RATIO


def _has_strong_native_text(text: str) -> bool:
    """Return True when a page clearly has a usable embedded text layer."""
    stripped = text.strip()
    if not stripped or _is_garbled_native_text(stripped):
        return False

    words = re.findall(r"\b\w+\b", stripped)
    return len(stripped) >= _NATIVE_TEXT_STRONG_CHARS or len(words) >= _NATIVE_TEXT_STRONG_WORDS


def _extract_native_paragraphs(
    page: fitz.Page,
    page_num: int = 1,
    header_pct: float = 0.08,
    footer_pct: float = 0.05,
) -> str:
    """Extract text from a native PDF page with proper paragraph breaks.

    ``page.get_text('text')`` inserts a ``\n`` after every line in the PDF,
    even when those lines are just wrapped lines of a single paragraph.
    This function uses block-level extraction to group visual paragraphs
    correctly, joins wrapped lines within a paragraph, and handles
    hyphenation.

    Running headers (top margin) are stripped from page 2 onward so the
    first-page title is preserved. Running footers (bottom margin) are
    stripped from all pages.
    """
    from config import settings  # local — avoids circular import on cold start

    page_height = page.rect.height

    # Use configured margins if not explicitly overridden
    if header_pct < 0:
        header_pct = settings.native_strip_header_pct
    if footer_pct < 0:
        footer_pct = settings.native_strip_footer_pct

    # Page 1 keeps the top margin (title/heading); pages 2+ strip headers
    strip_top = page_num > 1 and header_pct > 0.0
    strip_bottom = footer_pct > 0.0

    header_threshold = page_height * header_pct
    footer_threshold = page_height * (1.0 - footer_pct)

    blocks = page.get_text("blocks")
    paragraphs: list[str] = []

    for block in blocks:
        x0, y0, x1, y1, block_text = block[:5]

        # Skip header blocks (entirely in top margin)
        if strip_top and y1 < header_threshold:
            continue

        # Skip footer blocks (start in bottom margin)
        if strip_bottom and y0 > footer_threshold:
            continue

        if not block_text or not block_text.strip():
            continue

        lines = block_text.split("\n")
        para_lines: list[str] = []

        for line in lines:
            line = line.rstrip()
            if not line:
                # Empty line inside a block = explicit paragraph break
                if para_lines:
                    paragraphs.append(" ".join(para_lines))
                    para_lines = []
                continue

            if para_lines and para_lines[-1].endswith("-"):
                # Hyphenation: "manage-\nment" → "management"
                para_lines[-1] = para_lines[-1][:-1] + line.lstrip()
            else:
                para_lines.append(line)

        if para_lines:
            paragraphs.append(" ".join(para_lines))

    return "\n\n".join(paragraphs)


# ── Per-page worker (runs inside ProcessPoolExecutor child process) ────────

def process_single_page(
    page_data: Tuple[int, str, str, dict],
    engine_instance: Optional[Any] = None,
) -> Tuple[int, str, bool, Optional[dict]]:
    """Run OCR on a single page inside an isolated worker process.

    ``page_data`` is ``(page_num, pdf_path_str, engine_type, engine_kwargs)``.

    The worker process extracts the page bytes:
    * **Raw single-page PDF bytes** when ``engine_type == "google_docai"``
      and ``engine_kwargs["native_pdf"] == True``.  Passed directly to
      :meth:`ocr.google_engine.GoogleDocAIEngine.extract_from_pdf`.
    * **PNG image bytes** for all other engines (Paddle, Remote, legacy
      DocAI image path).  Decoded to ``PIL.Image`` before OCR.

    Returns ``(page_num, flat_text, success, structured_dict_or_None)``.
    """
    sys.setrecursionlimit(5000)  # PaddlePaddle's import chain is deep on Win.

    page_num, pdf_path_str, engine_type, engine_kwargs = page_data
    apply_tamil = bool(engine_kwargs.get("_normalize_tamil", True))
    use_layout = bool(engine_kwargs.get("use_layout", False))
    job_id_str = engine_kwargs.get("_job_id", "")
    is_docai = engine_type == "google_docai"
    is_docai_native = is_docai and bool(engine_kwargs.get("native_pdf", True))
    is_paddle_vl = engine_type in ("paddle_vl", "PaddleOCR1.5")
    is_olm_ocr = engine_type == "olm_ocr"
    is_vlm = is_paddle_vl or is_olm_ocr

    # Strip private helper kwargs + VLM-incompatible kwargs before forwarding.
    forwarded_kwargs = {
        k: v for k, v in engine_kwargs.items()
        if not k.startswith("_")
        # VLM engines do not accept lang, preprocess_options, etc.
        and not (is_vlm and k in ("lang", "preprocess_options",
                                  "enable_translation",
                                  "translation_target_lang",
                                  "use_layout", "layout_recovery",
                                  "table_as_html"))
    }

    try:
        import fitz
        import io
        from PIL import Image

        payload_bytes: bytes
        is_pdf_payload = False

        is_local = engine_type in ("sur_ocr", "paddle")
        local_image = None

        if is_local:
            from config import settings as _cfg
            dpi = _cfg.dpi_for_language(engine_kwargs.get("lang"))
            with fitz.open(pdf_path_str) as pdoc:
                pg = pdoc[page_num - 1]
                pix = pg.get_pixmap(dpi=dpi)
                local_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            payload_bytes = b""
        else:
            if is_docai_native:
                # Try single-page PDF slice first (best DocAI quality).
                # For text/hybrid PDFs, garbage=4 strips unused objects and the
                # slice fits under 20 MB. For purely scanned PDFs the full-res
                # embedded image IS the referenced object so the slice stays as
                # large as the whole PDF. We detect that and fall back to JPEG.
                single_page_doc = fitz.open()
                parent_doc = fitz.open(pdf_path_str)
                single_page_doc.insert_pdf(
                    parent_doc, from_page=page_num - 1, to_page=page_num - 1
                )
                buf = io.BytesIO()
                single_page_doc.save(buf, garbage=4, deflate=True)
                single_page_doc.close()
                parent_doc.close()

                candidate = buf.getvalue()
                del buf  # release buffer ASAP

                # DocAI online API hard limit: 20 MB per request.
                # We use 19 MB as the safety threshold.
                if len(candidate) < 19 * 1024 * 1024:
                    payload_bytes = candidate
                    is_pdf_payload = True
                    logger.debug(
                        "Page %d: PDF slice %d B accepted -> extract_from_pdf",
                        page_num, len(payload_bytes),
                    )
                else:
                    logger.debug(
                        "Page %d PDF slice too large for DocAI (%d B): using JPEG render.",
                        page_num, len(candidate),
                    )
                    del candidate  # free ~150 MB immediately

            if not is_pdf_payload:
                # JPEG image render fallback for large scanned pages.
                # JPEG compression reduces a high-res scanned page from ~150 MB
                # (raw PDF slice) to < 5 MB, well within DocAI limits.
                # We try progressively lower quality until size fits.
                from config import settings
                dpi = settings.dpi_for_language(engine_kwargs.get("lang"))
                with fitz.open(pdf_path_str) as pdoc:
                    pg = pdoc[page_num - 1]
                    pix = pg.get_pixmap(dpi=dpi)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                for jpeg_quality in (90, 75, 60, 45):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    payload_bytes = buf.getvalue()
                    size_mb = len(payload_bytes) / (1024 * 1024)
                    if len(payload_bytes) < 19 * 1024 * 1024:
                        logger.debug(
                            "Page %d: JPEG %.1f MB @ quality=%d -> DocAI image path",
                            page_num, size_mb, jpeg_quality,
                        )
                        break
                    logger.debug(
                        "Page %d: JPEG @ quality=%d too large (%.1f MB), reducing",
                        page_num, jpeg_quality, size_mb,
                    )

        # Create engine after determining payload type (or use override).
        engine = engine_instance or OCRFactory.get_engine(engine_type, **forwarded_kwargs)
        engine.initialize()

        def _apply_normalize_and_clean(page_result):
            """Normalize all element texts in-place."""
            for element in page_result.elements:
                element.text = normalize(
                    element.text or "", apply_tamil=apply_tamil
                )
            if is_llm_clean_enabled():
                el_dicts = [e.to_dict() for e in page_result.elements]
                llm_clean_elements(el_dicts)
                for el_dict, el_obj in zip(el_dicts, page_result.elements):
                    el_obj.text = el_dict.get("text", el_obj.text)
            return page_result

        # Path 1: DocAI PDF bytes (text/hybrid PDFs, highest quality).
        if is_pdf_payload and hasattr(engine, "extract_from_pdf"):
            page_result = engine.extract_from_pdf(
                pdf_bytes=payload_bytes, page_num=page_num
            )
            page_result.page_num = page_num
            _apply_normalize_and_clean(page_result)
            page_dict = page_result.to_dict()
            
            # Save rendered page image for preview (visual editor parity)
            try:
                from config import settings as _cfg
                pages_dir = Path(_cfg.output_dir) / job_id_str / "pages"
                pages_dir.mkdir(parents=True, exist_ok=True)
                img_path = pages_dir / f"page_{page_num}.png"
                
                dpi = _cfg.dpi_for_language(engine_kwargs.get("lang"))
                with fitz.open(pdf_path_str) as parent_doc:
                    pg = parent_doc[page_num - 1]
                    pix = pg.get_pixmap(dpi=dpi)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    img.save(str(img_path), "PNG")
                page_dict["image"] = f"pages/page_{page_num}.png"
                
                # Make sure to populate dimensions if not already present
                if "width" not in page_dict or not page_dict["width"]:
                    page_dict["width"] = page_result.width or pix.width
                if "height" not in page_dict or not page_dict["height"]:
                    page_dict["height"] = page_result.height or pix.height
            except Exception as img_exc:
                logger.warning("Failed to save DocAI page image: %s", img_exc)

            flat_text = page_result.to_plain_text()
            if not engine_instance:
                engine.cleanup()
            return page_num, flat_text, True, page_dict

        # Path 2: DocAI image path (JPEG fallback for large scanned pages).
        # DocAI ALWAYS produces rich structured output (paragraphs, tables,
        # headings) even when given an image. We MUST call extract_structured()
        # so the full PageResult is populated with elements. The `use_layout`
        # flag only controls non-DocAI engines (Paddle, Remote).
        if is_docai and hasattr(engine, "extract_structured"):
            image = Image.open(io.BytesIO(payload_bytes))
            page_result = engine.extract_structured(image)
            page_result.page_num = page_num
            _apply_normalize_and_clean(page_result)
            page_dict = page_result.to_dict()
            
            # Save rendered page image for preview (visual editor parity)
            try:
                from config import settings as _cfg
                pages_dir = Path(_cfg.output_dir) / job_id_str / "pages"
                pages_dir.mkdir(parents=True, exist_ok=True)
                img_path = pages_dir / f"page_{page_num}.png"
                image.save(str(img_path), "PNG")
                page_dict["image"] = f"pages/page_{page_num}.png"
                
                if "width" not in page_dict or not page_dict["width"]:
                    page_dict["width"] = page_result.width or image.width
                if "height" not in page_dict or not page_dict["height"]:
                    page_dict["height"] = page_result.height or image.height
            except Exception as img_exc:
                logger.warning("Failed to save DocAI page image: %s", img_exc)

            flat_text = page_result.to_plain_text()
            if not engine_instance:
                engine.cleanup()
            return page_num, flat_text, True, page_dict

        # Path 3: VLM Engines (PaddleVL / olmOCR) — renders page image and runs VLM inference.
        # extract_page() returns a LayoutPage (layout_schema) directly.
        if is_paddle_vl or is_olm_ocr:
            from config import settings as _cfg
            dpi = _cfg.ocr_dpi
            with fitz.open(pdf_path_str) as pdoc:
                pg = pdoc[page_num - 1]
                pix = pg.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            # We don't need a new instance if one was provided.
            if not engine_instance:
                engine = OCRFactory.get_engine(engine_type, **forwarded_kwargs)
                engine.initialize()
            
            # For olmOCR, pass the pdf_path and page number so it can perform document anchoring
            if is_olm_ocr:
                layout_page = engine.extract_page(img, page_number=page_num, pdf_path=pdf_path_str)
            else:
                layout_page = engine.extract_page(img, page_number=page_num)

            # Normalise block texts
            for blk in layout_page.blocks:
                blk.text = normalize(blk.text or "", apply_tamil=True)
            if is_llm_clean_enabled():
                for blk in layout_page.blocks:
                    blk.text = llm_clean_text(blk.text)
            flat_text = "\n\n".join(
                blk.text for blk in layout_page.blocks if blk.text
            )
            page_dict = layout_page.model_dump()
            
            # Save rendered page image for preview (visual editor parity)
            try:
                pages_dir = Path(_cfg.output_dir) / job_id_str / "pages"
                pages_dir.mkdir(parents=True, exist_ok=True)
                img_path = pages_dir / f"page_{page_num}.png"
                img.save(str(img_path), "PNG")
                page_dict["image"] = f"pages/page_{page_num}.png"
            except Exception as img_exc:
                logger.warning("Failed to save page preview image: %s", img_exc)

            if not engine_instance:
                engine.cleanup()
            return page_num, flat_text, True, page_dict

        # Path 4: Non-DocAI engines (Paddle / Remote / Surya).
        if is_local and local_image is not None:
            image = local_image
        else:
            image = Image.open(io.BytesIO(payload_bytes))

        if use_layout and hasattr(engine, "extract_structured"):
            page_result = engine.extract_structured(image)
            page_result.page_num = page_num
            _apply_normalize_and_clean(page_result)
            page_dict = page_result.to_dict()
            # Save rendered page image for preview (Surya and Paddle visual editor parity)
            if engine_type in ("sur_ocr", "paddle", "ocr_open_ai"):
                try:
                    from config import settings as _cfg
                    pages_dir = Path(_cfg.output_dir) / job_id_str / "pages"
                    pages_dir.mkdir(parents=True, exist_ok=True)
                    img_path = pages_dir / f"page_{page_num}.png"
                    image.save(str(img_path), "PNG")
                    page_dict["image"] = f"pages/page_{page_num}.png"
                    
                    if "width" not in page_dict or not page_dict["width"]:
                        page_dict["width"] = image.width
                    if "height" not in page_dict or not page_dict["height"]:
                        page_dict["height"] = image.height
                except Exception as img_exc:
                    logger.warning("Failed to save page image: %s", img_exc)
            flat_text = page_result.to_plain_text()
            if not engine_instance:
                engine.cleanup()
            return page_num, flat_text, True, page_dict

        text = normalize(
            engine.extract_text(image) or "", apply_tamil=apply_tamil
        )
        if is_llm_clean_enabled():
            text = llm_clean_text(text)
        if not engine_instance:
            engine.cleanup()
        return page_num, text, True, None

    except Exception as exc:  # noqa: BLE001 -- error envelope reported to caller
        logger.error("Failed to process page %d: %s", page_num, exc)
        return page_num, f"[ERROR: {exc}]", False, None

# ── PDFWorker class ────────────────────────────────────────────────────────

class PDFWorker:
    """Coordinator for OCR-ing one PDF end-to-end."""

    def __init__(
        self,
        job_id: str,
        pdf_path: Path,
        engine_type: str,
        engine_kwargs: dict,
        storage: Any,
        progress_tracker: ProgressTracker,
        resume_handler: ResumeHandler,
        max_workers: Optional[int] = None,
        doc_type: str = "auto",
    ) -> None:
        self.job_id = job_id
        self.pdf_path = pdf_path
        self.engine_type = engine_type
        self.engine_kwargs = dict(engine_kwargs)
        self.storage = storage
        self.progress_tracker = progress_tracker
        self.resume_handler = resume_handler
        self.doc_type = doc_type
        self.max_workers = self._resolve_max_workers(max_workers)

    @staticmethod
    def _resolve_max_workers(max_workers: Optional[int]) -> int:
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            return max(1, cpu_count - 1)
        return max(1, max_workers)

    # ── Auto-detect doc type (scanned vs native) ──────────────────────────

    def _auto_detect_doc_type(self) -> str:
        """Return ``'scanned'`` or ``'native'`` by sampling the first N pages.

        Signals (each adds weight toward ``'scanned'``):
        1. **File size** — very large per-page bytes suggest embedded images.
        2. **Images on page** — presence of raster images is strong scanned signal.
        3. **Low text density** — text area is tiny compared to total page area.
        4. **Garbled text** — mixed alphanumeric noise typical of OCR text layer.
        5. **Filename hint** — "Adobe Scan", "CamScanner", etc.

        A cumulative score >= ``_SCANNED_SCORE_THRESHOLD`` returns ``'scanned'``.
        """
        file_size = self.pdf_path.stat().st_size
        with fitz.open(self.pdf_path) as doc:
            total_pages = len(doc)
            avg_bytes_per_page = file_size / max(total_pages, 1)

            score = 0
            reason_parts: list[str] = []
            strong_native_pages = 0
            garbled_pages = 0

            # 1. File-size heuristic
            if avg_bytes_per_page > _SCANNED_SIZE_THRESHOLD_BYTES_PER_PAGE:
                score += 2
                reason_parts.append(
                    f"large file ({avg_bytes_per_page:,.0f} B/page)"
                )

            # 2. Filename heuristic
            name = self.pdf_path.name
            if _SCANNED_FILENAME_RE.search(name):
                score += _FILENAME_HINT_WEIGHT
                reason_parts.append("filename scan hint")

            # 3. Page-level probes
            probe_pages = min(total_pages, _DOC_TYPE_PROBE_PAGES)
            for page_idx in range(probe_pages):
                page = doc[page_idx]
                page_area = page.rect.width * page.rect.height
                if page_area <= 0:
                    continue

                raw_text = page.get_text("text") or ""
                has_strong_native_text = _has_strong_native_text(raw_text)
                if has_strong_native_text:
                    strong_native_pages += 1
                    reason_parts.append(
                        f"page {page_idx + 1} has strong text layer"
                    )

                # Images present?
                if page.get_images() and not has_strong_native_text:
                    score += _IMAGE_WEIGHT
                    reason_parts.append(f"page {page_idx + 1} has images")

                # Text density (get_text("blocks") returns text blocks only)
                blocks = page.get_text("blocks")
                text_area = sum(
                    (b[2] - b[0]) * (b[3] - b[1]) for b in blocks
                )
                density = text_area / page_area
                if density < 0.05 and not has_strong_native_text:  # Less than 5% text area
                    score += _LOW_TEXT_DENSITY_WEIGHT
                    reason_parts.append(
                        f"page {page_idx + 1} low text density ({density:.1%})"
                    )

                # Garbled text layer
                if _is_garbled_native_text(raw_text):
                    garbled_pages += 1
                    score += _GARBLED_TEXT_WEIGHT
                    reason_parts.append(
                        f"page {page_idx + 1} garbled text layer"
                    )

            # If probe pages clearly expose a usable text layer, prefer native.
            if strong_native_pages > 0 and garbled_pages < strong_native_pages:
                logger.info(
                    "Auto-detect %s → native (strong_native_pages=%d/%d, reasons=%s)",
                    self.pdf_path.name,
                    strong_native_pages,
                    probe_pages,
                    "; ".join(reason_parts) if reason_parts else "clean text layer",
                )
                return "native"

            reason = "; ".join(reason_parts) if reason_parts else "clean text layer"
            detected = (
                "scanned" if score >= _SCANNED_SCORE_THRESHOLD else "native"
            )
            logger.info(
                "Auto-detect %s -> %s (score=%d, reasons=%s)",
                self.pdf_path.name,
                detected,
                score,
                reason,
            )
            return detected

    # ── Public entry point ────────────────────────────────────────────────

    def process_pdf(self) -> None:
        """Process the entire PDF using the hybrid native / OCR pipeline."""
        try:
            # Resolve "auto" doc_type before touching any pages.
            if self.doc_type == "auto":
                detected = self._auto_detect_doc_type()
                if detected != "auto":
                    logger.info(
                        "Job %s doc_type auto-detected as '%s'",
                        self.job_id,
                        detected,
                    )
                    self.doc_type = detected
                    self.progress_tracker.update_doc_type(self.job_id, detected)

            logger.info("Starting PDF processing for job %s (type=%s)", self.job_id, self.doc_type)
            # Pass job_id so process_single_page can save page images.
            self.engine_kwargs["_job_id"] = self.job_id

            if self.doc_type == "native":
                from core.native_pipeline import try_run_native_pipeline
                logger.info("Job %s doc_type is native — invoking layout-aligned native extraction pipeline", self.job_id)
                self.progress_tracker.start_job(self.job_id)
                success = try_run_native_pipeline(
                    job_id=self.job_id,
                    pdf_path=self.pdf_path,
                    storage=self.storage,
                    progress_tracker=self.progress_tracker,
                )
                if success:
                    if self.engine_kwargs.get("use_layout", False):
                        self._build_combined_outputs()
                    logger.info("PDF processing completed via native pipeline for job %s", self.job_id)
                    return
                else:
                    logger.warning("Native PDF pipeline failed or was skipped; falling back to parallel OCR pipeline.")

            with fitz.open(self.pdf_path) as pdf_doc:
                total_pages = len(pdf_doc)
                processed_pages = self.resume_handler.get_processed_pages(self.job_id)
                if processed_pages:
                    logger.info(
                        "Resuming job %s: %d/%d pages already processed",
                        self.job_id,
                        len(processed_pages),
                        total_pages,
                    )

                self.progress_tracker.start_job(self.job_id)
                tasks = self._build_tasks(pdf_doc, total_pages, processed_pages)

            self._maybe_autodetect_language(tasks)
            # Re-stamp task kwargs with the now-resolved language so that
            # pre-built tasks never carry "auto" into process_single_page.
            tasks = self._restamp_task_langs(tasks)
            if tasks:
                self._run_ocr_in_parallel(tasks, total_pages)

            if self.engine_kwargs.get("use_layout", False):
                self._build_combined_outputs()

            logger.info("PDF processing completed for job %s", self.job_id)
        except Exception as exc:
            error_msg = f"PDF processing failed: {exc}"
            logger.error(error_msg)
            self.progress_tracker.mark_failed(self.job_id, error_msg)
            raise

    # ── Task building ─────────────────────────────────────────────────────

    def _build_tasks(
        self,
        pdf_doc: fitz.Document,
        total_pages: int,
        processed_pages: set,
    ) -> List[Tuple[int, bytes, str, dict]]:
        tasks: List[Tuple[int, bytes, str, dict]] = []
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            if page_num in processed_pages:
                continue

            page = pdf_doc[page_idx]
            if self._try_native_text(page, page_num, total_pages):
                continue

            tasks.append(self._render_ocr_task(page_num))
        return tasks

    def _try_native_text(self, page: fitz.Page, page_num: int, total_pages: int) -> bool:
        if self.doc_type not in ("auto", "native"):
            return False
        text = _extract_native_paragraphs(page, page_num=page_num)
        if not (self.doc_type == "native" or len(text.strip()) > NATIVE_TEXT_THRESHOLD):
            return False

        # Always check for garbled text, regardless of doc_type.
        #
        # Critical: A bilingual PDF can have doc_type="native" (because the
        # first 1-3 pages are clean English) while many subsequent pages use
        # legacy Tamil fonts (Bamini / TAB / TSCII) whose text layer is
        # Latin-1-encoded mojibake.  We must reject those pages and send them
        # to the OCR engine (Google DocAI) to get the real text.
        if _is_garbled_native_text(text):
            logger.info(
                "Page %d/%d: native text rejected (garbled / legacy font encoding) "
                "— routing to OCR engine",
                page_num, total_pages,
            )
            return False

        normalized = normalize(text, apply_tamil=self._tamil_normalize_for_text(text))
        # LLM-based cleaning for native text (fixes embedded OCR noise)
        if is_llm_clean_enabled():
            normalized = llm_clean_text(normalized)
        page_dict = {
            "text": normalized,
            "page": page_num,
            "method": "native",
            "elements": [{"type": "paragraph", "text": normalized}],
        }
        self.storage.append_text(self.job_id, normalized, page_num)
        if self.engine_kwargs.get("use_layout", False):
            self.storage.save_page_structure(self.job_id, page_num, page_dict)
        self.progress_tracker.update_progress(self.job_id, page_num, True)
        logger.info("Page %d/%d: native text extraction", page_num, total_pages)
        return True

    def _render_ocr_task(
        self,
        page_num: int,
    ) -> Tuple[int, str, str, dict]:
        """Build an OCR task for one PDF page.
        
        We pass the PDF path and page number so the worker process can do the
        heavy lifting (PDF slicing / PNG rendering) in parallel, rather than
        blocking the main thread loop.
        """
        return (
            page_num,
            str(self.pdf_path),
            self.engine_type,
            self._engine_kwargs_for_task(),
        )

    def _engine_kwargs_for_task(self) -> dict:
        kwargs = dict(self.engine_kwargs)
        kwargs.setdefault("_normalize_tamil", True)
        return kwargs

    @staticmethod
    def _tamil_normalize_for_text(text: str) -> bool:
        return is_predominantly_tamil(text)

    def _restamp_task_langs(self, tasks: list) -> list:
        """Return *tasks* with the ``lang`` kwarg updated to the resolved value.

        Tasks are pre-built before :meth:`_maybe_autodetect_language` runs, so
        they may carry ``"auto"`` as the language.  This method re-stamps every
        task with the language that was resolved (or detected) afterwards,
        ensuring ``process_single_page`` never receives ``"auto"``.

        Skipped for ``paddle_vl`` — the VLM handles language internally.
        """
        if self.engine_type in ("paddle_vl", "PaddleOCR1.5"):
            return tasks
        resolved = self.engine_kwargs.get("lang") or "en"
        if resolved == "auto":
            resolved = "en"
        return [(pn, img, et, {**kw, "lang": resolved}) for pn, img, et, kw in tasks]

    # ── Optional language auto-detect ─────────────────────────────────────

    def _maybe_autodetect_language(self, tasks: list) -> None:
        # paddle_vl is a VLM — it understands language from the image itself.
        # Language probing would try to pass lang= to PaddleVLEngine which
        # doesn't accept it.  Skip entirely.
        if self.engine_type in ("paddle_vl", "PaddleOCR1.5", "sur_ocr"):
            logger.info("%s: skipping language auto-detection", self.engine_type)
            return
        current_lang = self.engine_kwargs.get("lang", "auto")
        if not tasks or current_lang != "auto":
            if tasks:
                logger.info("Using explicit OCR language: %s", current_lang)
            return

        logger.info("Auto-detecting language from first %d pages", _LANG_PROBE_PAGE_COUNT)
        probe = [(t[0], t[1], t[2], t[3]) for t in tasks[:_LANG_PROBE_PAGE_COUNT]]
        detected = self._detect_language(probe)
        self.engine_kwargs["lang"] = detected
        self.progress_tracker.update_language(self.job_id, detected)
        logger.info("Detected OCR language: %s", detected)

    def _detect_language(self, page_data_list: List[Tuple[int, str, str, dict]]) -> str:
        """Probe up to N pages with a Tamil-capable OCR; conclude lang."""
        if not page_data_list:
            return "en"
        try:
            probe_engine = OCRFactory.get_engine(
                self.engine_type,
                **self._language_probe_kwargs(),
            )
            probe_engine.initialize()

            tamil_total = 0
            for page_data in page_data_list[:_LANG_PROBE_PAGE_COUNT]:
                page_num, pdf_path_str, _, _ = page_data
                
                # Render the page to image for probing
                from config import settings
                dpi = settings.dpi_for_language("ta")  # use target lang dpi
                with fitz.open(pdf_path_str) as parent_doc:
                    page = parent_doc[page_num - 1]
                    pix = page.get_pixmap(dpi=dpi)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                
                text = probe_engine.extract_text(img)
                tamil_total += len(_TAMIL_RE.findall(text))
                if tamil_total > _TAMIL_DETECT_THRESHOLD:
                    break
            probe_engine.cleanup()

            return "ta" if tamil_total > _TAMIL_DETECT_THRESHOLD else "en"
        except Exception as exc:
            logger.error("Language detection failed: %s", exc)
            return "en"

    def _language_probe_kwargs(self) -> dict:
        kwargs = {**self.engine_kwargs, "lang": "ta"}
        kwargs.pop("_normalize_tamil", None)
        return kwargs

    # ── Parallel OCR execution ────────────────────────────────────────────

    def _run_ocr_in_parallel(self, tasks: list, total_pages: int) -> None:
        # Remote engine doesn't need process isolation but we keep the same
        # path so OCR errors stay contained. GPU mode forces single worker.
        max_workers = 1 if self.engine_kwargs.get("use_gpu") else self.max_workers
        logger.info("OCR-ing %d pages with %d worker(s)", len(tasks), max_workers)

        mp_context = multiprocessing.get_context("spawn")
        sorted_pages = sorted(task[0] for task in tasks)
        cursor = 0
        buffer: dict = {}

        def flush() -> None:
            nonlocal cursor
            while cursor < len(sorted_pages):
                expected = sorted_pages[cursor]
                if expected not in buffer:
                    return
                text, success, page_dict = buffer.pop(expected)
                self.storage.append_text(self.job_id, text, expected)
                if page_dict is not None:
                    self.storage.save_page_structure(self.job_id, expected, page_dict)
                self.progress_tracker.update_progress(self.job_id, expected, success)
                if not success:
                    self.storage.log_error(
                        self.job_id, f"Page {expected} processing failed"
                    )
                # Log progress every 10 pages, plus always log page 1 and the last.
                if expected == 1 or expected == total_pages or expected % 10 == 0:
                    logger.info(
                        "OCR progress: page %d/%d (%.0f%%)",
                        expected, total_pages, 100.0 * expected / total_pages,
                    )
                else:
                    logger.debug("Wrote OCR page %d/%d", expected, total_pages)
                cursor += 1

        # =========================================================================
        # VLM Fast-Path (paddle_vl)
        # =========================================================================
        # A 16 GB VLM cannot be loaded per-page in a ProcessPoolExecutor (OOM).
        # It must be instantiated once in the main process and run sequentially.
        # =========================================================================
        if self.engine_type in ("paddle_vl", "PaddleOCR1.5", "sur_ocr"):
            logger.info(f"Running {self.engine_type} inference sequentially for {len(tasks)} pages")
            
            # Setup engine exactly once
            forwarded_kwargs = {
                k: v for k, v in self.engine_kwargs.items()
                if not k.startswith("_")
                and k not in ("lang", "preprocess_options", "enable_translation", 
                              "translation_target_lang", "use_layout", 
                              "layout_recovery", "table_as_html")
            }
            
            from ocr.factory import OCRFactory
            engine = OCRFactory.get_engine(self.engine_type, **forwarded_kwargs)
            
            try:
                engine.initialize()
                
                total_pages = len(tasks)
                logger.info(
                    f"[Job {self.job_id}] {self.engine_type}: starting sequential OCR for {total_pages} page(s)"
                )

                for idx, task in enumerate(tasks, start=1):
                    page_num = task[0]

                    # ── Check if job was cancelled/stopped before starting next page ──
                    pre_status = self.progress_tracker.get_status(self.job_id)
                    if pre_status.get("status") in ("cancelled", "failed"):
                        logger.info(
                            "[Job %s] Job status is '%s' — stopping OCR loop at page %d/%d",
                            self.job_id, pre_status.get("status"), idx, len(tasks),
                        )
                        break

                    logger.info(
                        f"[Job {self.job_id}] OCR page {idx}/{total_pages} (page_num={page_num}) — submitting"
                    )
                    try:
                        # Reuse the shared process_single_page logic but in the current thread
                        res_num, text, success, page_dict = process_single_page(
                            task, engine_instance=engine
                        )
                        logger.info(
                            f"[Job {self.job_id}] OCR page {idx}/{total_pages} returned: "
                            f"success={success} chars={len(text or '')} dict={'yes' if page_dict else 'no'}"
                        )
                        post_status = self.progress_tracker.get_status(self.job_id)
                        if post_status.get("status") in ("cancelled", "failed"):
                            logger.info(
                                "[Job %s] Discarding page %d result after cancellation",
                                self.job_id,
                                page_num,
                            )
                            break
                        buffer[page_num] = (text, success, page_dict)
                    except Exception as exc:
                        logger.error(
                            f"[Job {self.job_id}] OCR page {idx}/{total_pages} raised: {exc}",
                            exc_info=True,
                        )
                        buffer[page_num] = (f"[PAGE {page_num}: ERROR — {exc}]", False, None)

                    try:
                        flush()
                        current = self.progress_tracker.get_status(self.job_id)
                        logger.info(
                            f"[Job {self.job_id}] After flush: status={current.get('status')} "
                            f"processed={current.get('processed_pages')}/{total_pages}"
                        )
                    except Exception as flush_exc:
                        logger.error(
                            f"[Job {self.job_id}] flush() failed after page {page_num}: {flush_exc}",
                            exc_info=True,
                        )
                        # Surface to the job record so the UI doesn't stay "running" forever
                        self.progress_tracker.mark_failed(
                            self.job_id,
                            f"flush() failed after page {page_num}: {flush_exc}",
                        )
                        raise
            finally:
                try:
                    engine.cleanup()
                except Exception as cleanup_exc:
                    logger.warning(
                        f"[Job {self.job_id}] engine.cleanup() raised: {cleanup_exc}",
                        exc_info=True,
                    )

                # Defensive terminal-state reconciliation. update_progress() normally
                # auto-marks "completed" when processed_pages >= total_pages, but if
                # anything between request start and flush() failed silently, the job
                # could stay "running" forever. Surface that here so the UI never lies.
                try:
                    final = self.progress_tracker.get_status(self.job_id)
                    if final.get("status") == "running":
                        processed = final.get("processed_pages", 0)
                        if processed >= total_pages:
                            logger.info(
                                f"[Job {self.job_id}] All {total_pages} pages processed — reconciling status to 'completed'"
                            )
                            self.progress_tracker.update_progress(
                                self.job_id, total_pages, success=True
                            )
                        else:
                            logger.warning(
                                f"[Job {self.job_id}] OCR path ended with status=running "
                                f"({processed}/{total_pages} processed) — marking failed"
                            )
                            self.progress_tracker.mark_failed(
                                self.job_id,
                                f"OCR path ended without completing: {processed}/{total_pages} pages processed",
                            )
                except Exception as reconcile_exc:
                    logger.error(
                        f"[Job {self.job_id}] terminal-state reconciliation failed: {reconcile_exc}",
                        exc_info=True,
                    )

            return

        # =========================================================================
        # Standard Path (ProcessPoolExecutor)
        # =========================================================================
        # DocAI, VisionAI, and paddle-classic use the process pool.
        # =========================================================================
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            futures = {
                executor.submit(process_single_page, task): task[0] for task in tasks
            }
            jpeg_fallback_count = 0  # track pages that used image fallback

            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    res_num, text, success, page_dict = future.result(
                        timeout=PAGE_TIMEOUT_SECONDS
                    )
                    buffer[page_num] = (text, success, page_dict)
                except FutureTimeoutError:
                    logger.error("Page %d timed out after %ds", page_num, PAGE_TIMEOUT_SECONDS)
                    buffer[page_num] = (f"[PAGE {page_num}: TIMEOUT]", False, None)
                except Exception as exc:
                    logger.error("Error processing page %d: %s", page_num, exc)
                    buffer[page_num] = (f"[PAGE {page_num}: ERROR — {exc}]", False, None)
                flush()
            flush()

    # ── Post-processing ───────────────────────────────────────────────────

    def _build_combined_outputs(self) -> None:
        try:
            self.storage.build_combined_structured_result(self.job_id)
            if hasattr(self.storage, "build_html_result"):
                self.storage.build_html_result(self.job_id)
            
            # Post-process with layout-aware Markdown reconstruction for Surya OCR jobs
            if self.engine_type == "sur_ocr":
                try:
                    from scripts.surya_bbox_to_markdown import convert_file
                    import shutil
                    
                    result_json_path = self.storage.get_structured_result_path(self.job_id)
                    result_txt_path = self.storage.get_result_file_path(self.job_id)
                    result_md_path = self.storage.get_job_directory(self.job_id) / "result.md"
                    
                    if result_json_path.exists():
                        logger.info("Running refined layout-aware Markdown reconstruction for Surya job %s...", self.job_id)
                        # Overwrites result.txt (which gets displayed on UI results page)
                        convert_file(result_json_path, output_path=result_txt_path)
                        # Persist result.md for parity and direct downlaods
                        shutil.copy2(result_txt_path, result_md_path)
                        logger.info("Successfully reconstructed layout-aware Markdown and updated result.txt/result.md")
                except Exception as md_exc:
                    logger.error("Failed to generate layout-aware Markdown for job %s: %s", self.job_id, md_exc)

            logger.info("Built structured outputs for job %s", self.job_id)
        except Exception as exc:
            logger.warning("Failed to build combined structured output: %s", exc)
