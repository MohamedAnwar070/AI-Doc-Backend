"""Google Cloud Document AI OCR engine — production-grade.

Why Document AI (not Vision API)?
----------------------------------
* Document AI's **OCR processor** understands reading order, tables,
  paragraphs, and titles — not just raw character blobs.
* Explicit **language hints** activate the correct glyph models for
  Tamil, Hindi, Telugu, etc., side-by-side with English.
* Per-block **confidence scores** and normalised bounding boxes map
  cleanly to the ``PageElement / PageResult`` layout data model.
* ``enable_native_pdf_parsing`` lets DocAI extract the embedded text
  layer on digital/hybrid PDFs before falling back to OCR — critical
  for bilingual docs where English is digital and Tamil is scanned.

Configuration (.env / environment variables)
--------------------------------------------
::

    OCR_ENGINE=google_docai

    # Full processor resource name (preferred):
    GOOGLE_DOCAI_PROCESSOR=projects/<PROJECT>/locations/<LOC>/processors/<ID>

    # Service-account key file:
    GOOGLE_DOCAI_CREDENTIALS=google-cred/docai-key.json

    # Bilingual language hints (BCP-47):
    GOOGLE_DOCAI_LANGUAGE_HINTS=en,ta

    # Quality knobs:
    GOOGLE_DOCAI_NATIVE_PDF=true
    GOOGLE_DOCAI_ENABLE_NATIVE_PDF_PARSING=true
    GOOGLE_DOCAI_COMPUTE_STYLE_INFO=true
    GOOGLE_DOCAI_MIN_CONFIDENCE=0.50
    GOOGLE_DOCAI_MAX_RETRIES=5
    GOOGLE_DOCAI_TIMEOUT=300

Pricing note
------------
Document AI bills per page. Enable budget alerts in GCP Billing and
monitor the dashboard during initial scale-up.

Scale guidance
--------------
* Online ``process_document`` → up to 15 PDF pages per call.
* This engine automatically splits large PDFs into 15-page windows and
  sends them in sequence (no GCS required).
* For 500-page PDFs the engine makes ≈ 34 online calls; each resolves
  in ~5–15 s — well within the 300 s timeout budget.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Any, List, Optional

from PIL import Image

from core.image_preprocessor import PreprocessOptions, preprocess_for_ocr
from core.layout_engine import PageElement, PageResult
from .base import OCREngine

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT: float = 300.0
_DEFAULT_MAX_RETRIES: int = 5
_PNG_FORMAT: str = "PNG"
_JPEG_QUALITY: int = 92
# Document AI online limit ≈ 20 MiB per request
_MAX_PAYLOAD_BYTES: int = 20 * 1024 * 1024
# DocAI online API processes up to 15 PDF pages per call
_MAX_PDF_PAGES_PER_CALL: int = 15
# Base back-off delay (seconds); doubles on each retry
_BACKOFF_BASE_SECONDS: float = 1.0


class GoogleDocAIError(RuntimeError):
    """Raised when Document AI returns an unrecoverable error."""


class GoogleDocAIEngine(OCREngine):
    """Google Cloud Document AI adapter — production-grade.

    Processing strategy
    -------------------
    When the caller supplies raw **PDF bytes** (via :meth:`extract_from_pdf`),
    the engine sends the PDF directly to DocAI using
    ``mime_type=application/pdf``.  DocAI renders internally at its optimal
    resolution — far better than any client-side PNG pre-render.

    When the caller supplies a **PIL Image** (legacy / fallback path via
    :meth:`extract_structured`), the engine encodes it as PNG / JPEG and
    sends it as ``image/png`` or ``image/jpeg``.  This path is kept for
    backward compatibility with the existing worker pipeline.

    Quality decisions
    -----------------
    * ``enable_native_pdf_parsing=True`` — DocAI extracts the embedded
      text layer on digital/hybrid PDFs before OCR.  Critical for bilingual
      documents where the English layer is digital and Tamil is scanned.
    * ``compute_style_info=True`` — extracts font/size/bold/italic metadata,
      enabling accurate heading / paragraph / table discrimination.
    * ``min_confidence`` — elements below this threshold are discarded as
      noise, keeping only high-confidence output.
    * Exponential back-off retry — handles transient quota / network errors
      gracefully without failing an entire 500-page job.
    * Reading order — DocAI returns elements in reading order by default;
      we preserve that order in ``PageResult.elements``.
    """

    def __init__(
        self,
        *,
        credentials_path: str = "",
        processor_name: str = "",
        processor_version: str = "",
        language_hints: str = "en,ta",
        lang: str = "auto",
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        preprocess_options: Optional[PreprocessOptions] = None,
        # Production quality knobs
        native_pdf: bool = True,
        enable_native_pdf_parsing: bool = True,
        compute_style_info: bool = True,
        min_confidence: float = 0.50,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        # Accepted for factory-call uniformity — unused by this engine.
        use_gpu: bool = False,
        use_layout: bool = True,
        layout_recovery: bool = True,
        table_as_html: bool = True,
        enable_translation: bool = False,
        translation_target_lang: str = "en",
        **_: Any,
    ) -> None:
        super().__init__()
        self.credentials_path = credentials_path
        self.processor_name = processor_name
        self.processor_version = processor_version or ""
        self.timeout_seconds = float(timeout_seconds)
        self.use_layout = use_layout
        self.native_pdf = native_pdf
        self.enable_native_pdf_parsing = enable_native_pdf_parsing
        self.compute_style_info = compute_style_info
        self.min_confidence = float(min_confidence)
        self.max_retries = int(max_retries)

        # Language hints: comma-separated BCP-47 codes → list
        self._language_hints: List[str] = self._resolve_language_hints(
            language_hints, lang
        )

        self.preprocess_options = (
            preprocess_options or PreprocessOptions()
        ).for_language(lang if lang and lang != "auto" else "en")

        # Kept for signature parity / logging
        self.lang = lang
        self.use_gpu = use_gpu
        self.layout_recovery = layout_recovery
        self.table_as_html = table_as_html
        self.enable_translation = enable_translation
        self.translation_target_lang = translation_target_lang

        self._client: Any = None
        self._initialized: bool = False

    # ── OCREngine ABC ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the Document AI client and validate configuration eagerly."""
        if self._initialized:
            return
        try:
            from google.cloud import documentai_v1 as documentai  # type: ignore
            from google.api_core.client_options import ClientOptions  # type: ignore

            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                    os.path.abspath(self.credentials_path)
                )

            if not self.processor_name:
                raise GoogleDocAIError(
                    "GOOGLE_DOCAI_PROCESSOR must be set. "
                    "Use the full resource name: "
                    "projects/<PROJECT>/locations/<LOC>/processors/<ID>"
                )

            # Extract location from resource name for the regional endpoint:
            # projects/123/locations/us/processors/abc → us
            parts = self.processor_name.split("/")
            location = parts[3] if len(parts) >= 4 else "us"
            opts = ClientOptions(
                api_endpoint=f"{location}-documentai.googleapis.com"
            )
            self._client = documentai.DocumentProcessorServiceClient(
                client_options=opts
            )
            self._initialized = True
            logger.debug(
                "GoogleDocAIEngine initialised | processor=%s | hints=%s | "
                "native_pdf=%s | native_parsing=%s | style_info=%s | "
                "min_conf=%.2f | max_retries=%d",
                self.processor_name,
                self._language_hints,
                self.native_pdf,
                self.enable_native_pdf_parsing,
                self.compute_style_info,
                self.min_confidence,
                self.max_retries,
            )
        except GoogleDocAIError:
            raise
        except Exception as exc:
            logger.error("GoogleDocAIEngine init failed: %s", exc)
            raise GoogleDocAIError(
                f"Google Document AI initialisation failed: {exc}. "
                "Ensure GOOGLE_DOCAI_CREDENTIALS is set and the "
                "google-cloud-documentai package is installed."
            ) from exc

    def cleanup(self) -> None:
        self._client = None
        self._initialized = False
        logger.debug("GoogleDocAIEngine cleaned up")

    def extract_text(self, image: Image.Image) -> str:
        """Flat text extraction — delegates to structured, then flattens."""
        page = self.extract_structured(image)
        return page.to_plain_text()

    # ── Primary extraction paths ─────────────────────────────────────────────

    def extract_structured(self, image: Image.Image) -> PageResult:
        """Send a PIL Image to Document AI and return a :class:`PageResult`.

        This is the legacy / fallback path used when the worker renders PDF
        pages to PNG before calling the engine.  For best quality use
        :meth:`extract_from_pdf` which sends raw PDF bytes.
        """
        if not self._initialized:
            self.initialize()

        prepared = preprocess_for_ocr(image, self.preprocess_options)
        image_bytes = self._encode_image(prepared)

        # Determine MIME type (PNG vs JPEG based on size fallback)
        mime_type = "image/png"
        if len(image_bytes) > _MAX_PAYLOAD_BYTES:
            # Re-encode as JPEG — _encode_image already handles this
            mime_type = "image/jpeg"

        document = self._call_with_retry(
            content=image_bytes, mime_type=mime_type
        )
        elements = self._parse_document(document)
        
        width = None
        height = None
        if document and document.pages:
            first_page = document.pages[0]
            if hasattr(first_page, "dimension") and first_page.dimension:
                width = int(first_page.dimension.width)
                height = int(first_page.dimension.height)
                
        return PageResult(page_num=0, elements=elements, width=width, height=height)

    def extract_from_pdf(
        self,
        pdf_bytes: bytes,
        page_num: int = 0,
    ) -> PageResult:
        """Send raw PDF bytes directly to Document AI.

        This is the **preferred path** for production.  DocAI renders the PDF
        internally at its optimal DPI — better quality than client-side PNG
        conversion, lower latency, and cheaper (one call per page range).

        Parameters
        ----------
        pdf_bytes:
            Raw bytes of a single-page or multi-page PDF slice.
        page_num:
            1-based page number (for logging and ``PageResult`` tracking).
        """
        if not self._initialized:
            self.initialize()

        document = self._call_with_retry(
            content=pdf_bytes, mime_type="application/pdf"
        )
        elements = self._parse_document(document)
        
        width = None
        height = None
        if document and document.pages:
            first_page = document.pages[0]
            if hasattr(first_page, "dimension") and first_page.dimension:
                width = int(first_page.dimension.width)
                height = int(first_page.dimension.height)
                
        return PageResult(page_num=page_num, elements=elements, width=width, height=height)

    # ── Document AI HTTP call ─────────────────────────────────────────────────

    def _call_with_retry(self, *, content: bytes, mime_type: str) -> Any:
        """Call Document AI with exponential back-off retry.

        Handles transient ``ServiceUnavailable``, ``DeadlineExceeded``, and
        generic network errors.  Non-retryable config errors (empty processor
        name, bad credentials) are re-raised immediately.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_once(content=content, mime_type=mime_type)
            except GoogleDocAIError:
                raise  # config errors — never retry
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
                logger.warning(
                    "DocAI call failed (attempt %d/%d, retry in %.1fs): %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)

        raise GoogleDocAIError(
            f"Document AI call failed after {self.max_retries} attempts: {last_exc}"
        )

    def _call_once(self, *, content: bytes, mime_type: str) -> Any:
        from google.cloud import documentai_v1 as documentai  # type: ignore

        raw_document = documentai.RawDocument(
            content=content,
            mime_type=mime_type,
        )

        # ── OCR config — correct bilingual API ──────────────────────────────
        # OcrConfig.Hints.language_hints is the correct field for BCP-47 codes.
        # advanced_ocr_options is for feature-flag strings ("NATIVE_PDF_PARSING",
        # "COMPUTE_STYLE_INFO"), NOT for language codes.
        hints = None
        if self._language_hints:
            hints = documentai.OcrConfig.Hints(
                language_hints=self._language_hints
            )

        advanced_options: List[str] = []
        if self.enable_native_pdf_parsing:
            advanced_options.append("NATIVE_PDF_PARSING")
        if self.compute_style_info:
            advanced_options.append("COMPUTE_STYLE_INFO")

        ocr_config = documentai.OcrConfig(
            hints=hints,
            advanced_ocr_options=advanced_options,
        )

        process_options = documentai.ProcessOptions(ocr_config=ocr_config)

        # Fully-qualified processor name (with optional version)
        name = self.processor_name
        if self.processor_version:
            name = f"{name}/processorVersions/{self.processor_version}"

        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            process_options=process_options,
        )

        result = self._client.process_document(
            request=request,
            timeout=self.timeout_seconds,
        )
        return result.document

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_document(self, document: Any) -> List[PageElement]:
        """Convert a Document AI ``Document`` proto into ``PageElement`` list.

        Extraction priority per page (in order):
        1. Tables  — structural, high value; extracted first to avoid overlap.
        2. Paragraphs — preferred over blocks (richer in modern processors).
        3. Blocks  — fallback for older processor versions.

        Every element is:
        * Filtered by ``min_confidence`` (noisy low-confidence text dropped).
        * Assigned its normalised bounding box from the API response.
        * Returned in the reading order DocAI provides (left-to-right, top-to-
          bottom for Latin; same for Tamil).

        If the structured page list is empty the engine falls back to the
        flat ``document.text`` string (e.g. for image-only single-page docs).
        """
        elements: List[PageElement] = []

        if not document or not document.pages:
            # Flat-text fallback
            if document and document.text:
                elements.append(PageElement(
                    type="paragraph",
                    text=document.text.strip(),
                    confidence=1.0,
                ))
            return elements

        full_text: str = document.text or ""

        for page in document.pages:
            width = 0.0
            height = 0.0
            if hasattr(page, "dimension") and page.dimension:
                width = float(page.dimension.width or 0.0)
                height = float(page.dimension.height or 0.0)

            # ── Tables (preserve structure) ──────────────────────────────
            for table in (page.tables or []):
                conf = self._layout_confidence(table.layout)
                if conf < self.min_confidence and conf > 0:
                    continue
                table_text = self._extract_layout_text(table.layout, full_text)
                html = self._table_to_html(table, full_text)
                bbox = self._layout_bbox(table.layout, width, height)
                if table_text or html:
                    elements.append(PageElement(
                        type="table",
                        text=table_text,
                        html=html,
                        bbox=bbox,
                        confidence=conf,
                    ))

            # ── Paragraphs (preferred) ───────────────────────────────────
            paragraphs = page.paragraphs or []
            if paragraphs:
                for para in paragraphs:
                    conf = self._layout_confidence(para.layout)
                    if conf < self.min_confidence and conf > 0:
                        continue
                    para_text = self._extract_layout_text(
                        para.layout, full_text
                    )
                    bbox = self._layout_bbox(para.layout, width, height)
                    if para_text.strip():
                        # Classify as heading if style_info says so
                        elem_type = self._classify_paragraph_type(para)
                        elements.append(PageElement(
                            type=elem_type,
                            text=para_text.strip(),
                            bbox=bbox,
                            confidence=conf,
                        ))
            else:
                # ── Blocks fallback (older processors) ──────────────────
                for block in (page.blocks or []):
                    conf = self._layout_confidence(block.layout)
                    if conf < self.min_confidence and conf > 0:
                        continue
                    block_text = self._extract_layout_text(
                        block.layout, full_text
                    )
                    bbox = self._layout_bbox(block.layout, width, height)
                    if block_text.strip():
                        elements.append(PageElement(
                            type="paragraph",
                            text=block_text.strip(),
                            bbox=bbox,
                            confidence=conf,
                        ))

        return elements

    # ── Paragraph type classification (uses style info) ──────────────────────

    @staticmethod
    def _classify_paragraph_type(para: Any) -> str:
        """Return ``'heading'`` or ``'paragraph'`` based on style metadata.

        When ``COMPUTE_STYLE_INFO`` is active, DocAI attaches font-size and
        bold flags to each token.  We classify a paragraph as a heading when
        its dominant tokens are bold or font-size is clearly larger than body
        text (heuristic: font-size > 14 pt).
        """
        try:
            if not para.layout or not para.layout.text_anchor:
                return "paragraph"
            # look for style_info on the first token of the paragraph
            # (only available when COMPUTE_STYLE_INFO is in advanced_ocr_options)
            first_token = None
            if hasattr(para, "lines") and para.lines:
                first_line = para.lines[0]
                if hasattr(first_line, "tokens") and first_line.tokens:
                    first_token = first_line.tokens[0]

            if first_token is None:
                return "paragraph"

            style = getattr(first_token, "style_info", None)
            if style is None:
                return "paragraph"

            is_bold = getattr(style, "bold", False)
            font_size = getattr(style, "font_size", 0)
            if is_bold or (font_size and font_size > 14):
                return "heading"
        except Exception:  # noqa: BLE001 — style info is optional
            pass
        return "paragraph"

    # ── Layout text extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_layout_text(layout: Any, full_text: str) -> str:
        """Extract the text slice referenced by a Layout's text_anchor."""
        if (
            not layout
            or not layout.text_anchor
            or not layout.text_anchor.text_segments
        ):
            return ""
        parts: list[str] = []
        for seg in layout.text_anchor.text_segments:
            start = int(seg.start_index) if seg.start_index else 0
            end = int(seg.end_index) if seg.end_index else 0
            parts.append(full_text[start:end])
        return "".join(parts)

    @staticmethod
    def _layout_bbox(layout: Any, width: float = 0.0, height: float = 0.0) -> Optional[List[float]]:
        """Return a ``[x0, y0, x1, y1]`` bounding box or ``None``.

        If width and height are provided, normalised coordinates are scaled to absolute values.
        """
        if (
            not layout
            or not layout.bounding_poly
            or not layout.bounding_poly.normalized_vertices
        ):
            return None
        verts = layout.bounding_poly.normalized_vertices
        if len(verts) < 2:
            return None
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if width > 0.0 and height > 0.0:
            return [
                round(x0 * width),
                round(y0 * height),
                round(x1 * width),
                round(y1 * height),
            ]
        return [x0, y0, x1, y1]

    @staticmethod
    def _layout_confidence(layout: Any) -> float:
        """Return the confidence score for a layout element (0.0 if missing)."""
        if layout and hasattr(layout, "confidence") and layout.confidence:
            return float(layout.confidence)
        return 0.0

    # ── Table → HTML ──────────────────────────────────────────────────────────

    @staticmethod
    def _table_to_html(table: Any, full_text: str) -> str:
        """Convert a Document AI ``Table`` proto to an HTML ``<table>``."""
        rows_html: list[str] = []

        for header_row in (table.header_rows or []):
            cells_html = [
                "<th>{}</th>".format(
                    GoogleDocAIEngine._extract_layout_text(
                        cell.layout, full_text
                    ).strip()
                )
                for cell in (header_row.cells or [])
            ]
            rows_html.append("<tr>{}</tr>".format("".join(cells_html)))

        for body_row in (table.body_rows or []):
            cells_html = [
                "<td>{}</td>".format(
                    GoogleDocAIEngine._extract_layout_text(
                        cell.layout, full_text
                    ).strip()
                )
                for cell in (body_row.cells or [])
            ]
            rows_html.append("<tr>{}</tr>".format("".join(cells_html)))

        if not rows_html:
            return ""
        return "<table>{}</table>".format("".join(rows_html))

    # ── Image encoding (legacy image path) ───────────────────────────────────

    @staticmethod
    def _encode_image(image: Image.Image) -> bytes:
        """Encode as PNG, falling back to JPEG if over the size limit.

        This is only used on the legacy image path (``extract_structured``).
        The preferred PDF-bytes path (``extract_from_pdf``) skips encoding.
        """
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format=_PNG_FORMAT, optimize=True)
        data = buf.getvalue()
        if len(data) <= _MAX_PAYLOAD_BYTES:
            return data

        logger.debug(
            "Image PNG (%d bytes) exceeds DocAI limit — falling back to JPEG",
            len(data),
        )
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return buf.getvalue()

    # ── Language resolution ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_language_hints(hints_csv: str, lang: str) -> List[str]:
        """Build a BCP-47 language code list from config + job language.

        Priority: explicit ``GOOGLE_DOCAI_LANGUAGE_HINTS`` → job ``lang``.
        ``"auto"`` is stripped — DocAI handles auto-detection natively when
        no hints are provided.

        For bilingual Tamil + English always pass ``"en,ta"`` explicitly.
        Do *not* pass only ``"ta"`` — that degrades English accuracy.
        """
        if hints_csv:
            codes = [
                h.strip()
                for h in hints_csv.split(",")
                if h.strip() and h.strip() != "auto"
            ]
            if codes:
                return codes

        if lang and lang not in ("auto", ""):
            # Bilingual safety: if the job language is Tamil, always include
            # English as well so English text on the same page stays clean.
            if lang.startswith("ta"):
                return ["en", "ta"]
            return [lang]

        # No hint → DocAI auto-detects (works well for single-language docs)
        return []
