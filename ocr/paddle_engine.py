"""Local PaddleOCR engine.

Supports two modes:

* **Flat OCR** — single-pass text extraction. Used for Tamil pages
  (layout detection is unsafe for ``ta``).
* **PP-Structure layout** — region detection + table HTML, used for
  English / Chinese pages.

Hybrid mixed-language handling: when the document language differs from
the layout model's language, a separate flat OCR pass with the *correct*
language is used for the actual text, while PP-Structure provides only
the structural hints (bboxes / region types).

Quality plumbing
----------------
* All page images are pushed through the shared
  :func:`core.image_preprocessor.preprocess_for_ocr` pipeline before OCR.
* Spatial 2-column table reconstruction is delegated to
  :mod:`ocr._table_reconstructor`.

This file has been kept narrow to satisfy SonarQube cognitive-complexity
limits — pure logic lives elsewhere.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.image_preprocessor import PreprocessOptions, preprocess_for_ocr
from core.layout_engine import LayoutEngine, PageElement, PageResult, _LAYOUT_SUPPORTED_LANGS
from core.translator import detect_and_translate_elements

from ._table_reconstructor import reconstruct_table
from .base import OCREngine

logger = logging.getLogger(__name__)


# ── Tunables ───────────────────────────────────────────────────────────────

_DEFAULT_CPU_THREADS: int = 4
_LINE_GROUP_RATIO: float = 0.8
_LINE_MIN_THRESHOLD: float = 10.0


class PaddleEngine(OCREngine):
    """PaddleOCR adapter — flat OCR + PP-Structure with Tamil-aware fallbacks."""

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        use_layout: bool = False,
        layout_recovery: bool = True,
        table_as_html: bool = True,
        enable_translation: bool = False,
        translation_target_lang: str = "en",
        preprocess_options: Optional[PreprocessOptions] = None,
    ) -> None:
        super().__init__()
        self.use_gpu = use_gpu
        self.lang = lang
        self.use_layout = use_layout
        self.layout_recovery = layout_recovery
        self.table_as_html = table_as_html
        self.enable_translation = enable_translation
        self.translation_target_lang = translation_target_lang
        self.preprocess_options = preprocess_options or PreprocessOptions()

        self.ocr: Any = None
        self._en_ocr: Any = None
        self._layout_engine: Optional[LayoutEngine] = None
        self._layout_lang: Optional[str] = None
        self._initialized: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            self._disable_layout_for_tamil_if_needed()
            if self.use_layout:
                self._initialize_layout_mode()
            else:
                self._initialize_flat_mode()
            self._initialized = True
        except Exception as exc:
            logger.error("Failed to initialise PaddleOCR: %s", exc)
            raise

    def cleanup(self) -> None:
        if self.ocr is not None:
            del self.ocr
            self.ocr = None
        if self._en_ocr is not None:
            del self._en_ocr
            self._en_ocr = None
        if self._layout_engine is not None:
            self._layout_engine.cleanup()
            self._layout_engine = None
        self._initialized = False
        logger.info("PaddleOCR cleaned up")

    # ── Public extraction ─────────────────────────────────────────────────

    def extract_text(self, image: Image.Image) -> str:
        if not self._initialized:
            self.initialize()
        if self.use_layout and self._layout_engine is not None:
            return self.extract_structured(image).to_plain_text()
        return self._flat_extract_text(image)

    def extract_structured(self, image: Image.Image) -> PageResult:
        if not self._initialized:
            self.initialize()

        if self._layout_engine is None:
            return self._extract_via_flat_ocr_only(image)
        return self._extract_via_layout_pipeline(image)

    # ── Initialisation helpers ────────────────────────────────────────────

    def _disable_layout_for_tamil_if_needed(self) -> None:
        if self.lang == "ta" and self.use_layout:
            logger.warning(
                "Layout detection (PP-Structure) is disabled for Tamil ('ta') "
                "to prevent character garbling. Using high-fidelity flat OCR instead."
            )
            self.use_layout = False

    def _initialize_layout_mode(self) -> None:
        self._layout_lang = (
            self.lang if self.lang in _LAYOUT_SUPPORTED_LANGS else "en"
        )
        if self.lang not in _LAYOUT_SUPPORTED_LANGS:
            logger.warning(
                "PP-Structure layout mode does not support lang='%s' "
                "(only %s). Using layout_lang='%s' for layout and "
                "flat OCR with lang='%s' for text.",
                self.lang,
                sorted(_LAYOUT_SUPPORTED_LANGS),
                self._layout_lang,
                self.lang,
            )

        logger.info(
            "Initialising PP-Structure (GPU=%s, lang=%s)",
            self.use_gpu,
            self._layout_lang,
        )
        self._layout_engine = LayoutEngine(
            use_gpu=self.use_gpu,
            lang=self._layout_lang,
            recovery=self.layout_recovery,
            table_as_html=self.table_as_html,
        )
        self._layout_engine.initialize()

        if self.lang != self._layout_lang:
            self._initialize_auxiliary_flat_ocr()
            self._initialize_english_helper_ocr()

    def _initialize_flat_mode(self) -> None:
        logger.info("Initialising flat PaddleOCR (GPU=%s, lang=%s)", self.use_gpu, self.lang)
        self.ocr = self._build_paddle_ocr(self.lang)
        if self.lang == "ta":
            self._initialize_english_helper_ocr()

    def _initialize_auxiliary_flat_ocr(self) -> None:
        logger.info(
            "Initialising auxiliary flat PaddleOCR (GPU=%s, lang=%s) "
            "for accurate text extraction",
            self.use_gpu,
            self.lang,
        )
        self.ocr = self._build_paddle_ocr(self.lang)

    def _initialize_english_helper_ocr(self) -> None:
        logger.info("Initialising English flat PaddleOCR for date / number rescue")
        self._en_ocr = self._build_paddle_ocr("en")

    def _build_paddle_ocr(self, lang: str) -> Any:
        from paddleocr import PaddleOCR

        # "auto" is not a valid PaddleOCR lang code; fall back to English.
        safe_lang = lang if lang and lang != "auto" else "en"
        return PaddleOCR(
            use_angle_cls=True,
            lang=safe_lang,
            use_gpu=self.use_gpu,
            show_log=False,
            enable_mkldnn=not self.use_gpu,
            cpu_threads=_DEFAULT_CPU_THREADS if not self.use_gpu else 1,
        )

    # ── Flat OCR pathway ──────────────────────────────────────────────────

    def _flat_extract_text(self, image: Image.Image) -> str:
        ocr_items = self._run_flat_ocr_raw(image)
        if not ocr_items:
            return ""
        return "\n".join(self._reconstruct_lines(ocr_items))

    def _run_flat_ocr_raw(self, image: Image.Image) -> list:
        return self._run_named_ocr(self.ocr, image, "flat OCR")

    def _run_en_ocr_raw(self, image: Image.Image) -> list:
        return self._run_named_ocr(self._en_ocr, image, "English helper OCR")

    def _run_named_ocr(self, paddle_ocr: Any, image: Image.Image, label: str) -> list:
        if paddle_ocr is None:
            return []
        try:
            processed = preprocess_for_ocr(image, self.preprocess_options)
            result = paddle_ocr.ocr(np.array(processed), cls=True)
            if not result:
                return []
            return result[0] or []
        except Exception as exc:
            logger.error("%s failed: %s", label, exc)
            return []

    # ── Structured extraction pathways ────────────────────────────────────

    def _extract_via_flat_ocr_only(self, image: Image.Image) -> PageResult:
        """Used when PP-Structure is disabled (e.g. Tamil)."""
        logger.debug(
            "Layout engine disabled; reconstructing tables directly from flat OCR"
        )
        ocr_items = self._run_flat_ocr_raw(image)
        en_items = self._run_en_ocr_raw(image)
        tables, remaining, remaining_bbox = reconstruct_table(ocr_items, en_items)

        elements = self._assemble_elements(tables, remaining, remaining_bbox, ocr_items)
        page = PageResult(page_num=0, elements=elements)
        detect_and_translate_elements(
            page.elements,
            enable_translation=self.enable_translation,
            target_lang=self.translation_target_lang,
        )
        return page

    def _extract_via_layout_pipeline(self, image: Image.Image) -> PageResult:
        """Used when PP-Structure is initialised."""
        try:
            page = self._layout_engine.extract(image)
        except Exception as exc:
            logger.error("Structured extraction failed: %s", exc)
            return PageResult(
                page_num=0,
                elements=[PageElement(type="paragraph", text=f"[ERROR: {exc}]")],
            )

        if self._needs_hybrid_text_rebuild():
            self._rebuild_text_with_correct_language(page, image)

        detect_and_translate_elements(
            page.elements,
            enable_translation=self.enable_translation,
            target_lang=self.translation_target_lang,
        )
        return page

    def _needs_hybrid_text_rebuild(self) -> bool:
        return self.ocr is not None and self.lang != self._layout_lang

    def _rebuild_text_with_correct_language(
        self,
        page: PageResult,
        image: Image.Image,
    ) -> None:
        ocr_items = self._run_flat_ocr_raw(image)
        en_items = self._run_en_ocr_raw(image)
        tables, remaining, remaining_bbox = reconstruct_table(ocr_items, en_items)

        if tables:
            page.elements = self._assemble_elements(
                tables, remaining, remaining_bbox, ocr_items
            )
            return

        pp_table_bboxes = [e.bbox for e in page.elements if e.type == "table" and e.bbox]
        if pp_table_bboxes and ocr_items:
            page.elements = self._merge_pp_layout_with_flat_ocr(
                page.elements, ocr_items
            )
        else:
            flat_text = "\n".join(self._reconstruct_lines(ocr_items)) if ocr_items else ""
            page.elements = [PageElement(type="paragraph", text=flat_text)]

    @classmethod
    def _assemble_elements(
        cls,
        tables: list,
        remaining: str,
        remaining_bbox,
        ocr_items: list,
    ) -> list:
        elements: list[PageElement] = []
        if remaining and remaining.strip():
            elements.append(
                PageElement(type="paragraph", text=remaining, bbox=remaining_bbox)
            )
        if tables:
            elements.extend(tables)

        if not elements and ocr_items:
            flat_text = "\n".join(cls._reconstruct_lines(ocr_items))
            elements.append(PageElement(type="paragraph", text=flat_text))
        return elements

    # ── Layout-merge fallback when flat-OCR table reconstruction fails ────

    def _merge_pp_layout_with_flat_ocr(
        self,
        pp_elements: list,
        ocr_items: list,
    ) -> list:
        from ._table_reconstructor import _parse_items, _group_into_rows

        parsed = _parse_items(ocr_items)
        if not parsed:
            return pp_elements

        result: list[PageElement] = []
        assigned: set[int] = set()
        for el in pp_elements:
            if not el.bbox:
                result.append(el)
                continue
            region_text = self._region_text_from_parsed(parsed, el.bbox, assigned)
            if region_text is None:
                result.append(el)
                continue
            result.append(
                PageElement(
                    type=el.type,
                    text=region_text,
                    html=None,
                    bbox=el.bbox,
                    confidence=el.confidence,
                )
            )

        unassigned_idx = [i for i in range(len(parsed)) if i not in assigned]
        if unassigned_idx:
            unassigned_items = [parsed[i] for i in unassigned_idx]
            rows = _group_into_rows(unassigned_items)
            extra_text = "\n".join(
                " ".join(it.text for it in row) for row in rows
            ).strip()
            if extra_text:
                result.append(PageElement(type="paragraph", text=extra_text))
        return result

    @staticmethod
    def _region_text_from_parsed(
        parsed: list,
        bbox: list,
        assigned: set[int],
    ) -> Optional[str]:
        from ._table_reconstructor import _group_into_rows

        bx0, by0, bx1, by1 = bbox
        region_indices: list[int] = []
        for idx, item in enumerate(parsed):
            cx = (item.xl + item.xr) / 2.0
            cy = (item.yt + item.yb) / 2.0
            if bx0 - 20 <= cx <= bx1 + 20 and by0 - 20 <= cy <= by1 + 20:
                region_indices.append(idx)
        if not region_indices:
            return None
        for idx in region_indices:
            assigned.add(idx)
        region_items = [parsed[i] for i in region_indices]
        rows = _group_into_rows(region_items)
        return "\n".join(" ".join(it.text for it in row) for row in rows)

    # ── Line reconstruction (flat OCR helper) ─────────────────────────────

    @staticmethod
    def _reconstruct_lines(ocr_items: List) -> List[str]:
        grouped: List[Tuple[float, float, str]] = []
        for item in ocr_items:
            if not item or len(item) < 2:
                continue
            box = item[0]
            text_part = item[1]
            text = text_part[0] if text_part and len(text_part) >= 1 else ""
            if not text:
                continue
            try:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
            except (TypeError, IndexError):
                continue
            grouped.append(
                (
                    float((min(ys) + max(ys)) / 2.0),
                    float(min(xs)),
                    text,
                )
            )

        if not grouped:
            return []

        grouped.sort(key=lambda t: (t[0], t[1]))
        diffs = [
            abs(grouped[i][0] - grouped[i - 1][0]) for i in range(1, len(grouped))
        ]
        median_diff = float(np.median(diffs)) if diffs else _LINE_MIN_THRESHOLD
        y_threshold = max(_LINE_MIN_THRESHOLD, median_diff * _LINE_GROUP_RATIO)

        lines: List[List[Tuple[float, str]]] = []
        current: List[Tuple[float, str]] = []
        current_y: Optional[float] = None

        for y, x, text in grouped:
            if current_y is None or abs(y - current_y) <= y_threshold:
                current.append((x, text))
                current_y = y if current_y is None else (current_y * 0.7) + (y * 0.3)
            else:
                lines.append(current)
                current = [(x, text)]
                current_y = y
        if current:
            lines.append(current)

        out: List[str] = []
        for line in lines:
            line.sort(key=lambda t: t[0])
            joined = " ".join(text for _, text in line).strip()
            if joined:
                out.append(joined)
        return out
