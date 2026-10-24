"""Local Sur OCRengine (v0.17.x API).

Uses Surya's predictor classes for text detection + recognition.
Produces structured :class:`PageResult` with per-line bounding boxes
and confidence scores — identical contract to PaddleEngine / RemoteEngine.

Configuration via environment variables:
    ``OCR_ENGINE=sur_ocr``  — activates this engine.
    ``SURYA_DPI``           — PDF render DPI (default 200).
    ``SURYA_LANGS``         — comma-separated language codes (default "ta,en").
    ``GPU_ENABLED``         — enables CUDA if available.

Usage::

    engine = SuryaEngine(lang="ta,en")
    engine.initialize()
    page_result = engine.extract_structured(pil_image)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

from core.image_preprocessor import PreprocessOptions, preprocess_for_ocr
from core.layout_engine import PageElement, PageResult
from .base import OCREngine

logger = logging.getLogger(__name__)


class SuryaEngine(OCREngine):
    """Sur OCRadapter — detection + recognition with bounding boxes."""

    def __init__(
        self,
        lang: str = "ta,en",
        use_gpu: bool = False,
        preprocess_options: Optional[PreprocessOptions] = None,
        # Accepted for factory-call parity — unused by Surya.
        enable_translation: bool = False,
        translation_target_lang: str = "en",
        **_kwargs,
    ) -> None:
        super().__init__()
        self.lang = lang or "ta,en"
        self.use_gpu = use_gpu
        self.preprocess_options = preprocess_options or PreprocessOptions()

        # Model predictors — loaded once in initialize()
        self._foundation_predictor = None
        self._det_predictor = None
        self._rec_predictor = None
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            # ── CPU Optimizations for PyTorch/Surya ────────────────────────
            import os
            import torch
            
            # Prevent PyTorch thread over-subscription/contention on CPU
            try:
                torch.set_num_threads(1)
            except RuntimeError:
                pass
            try:
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            
            # Lower batch size on CPU to prevent cache/memory thrashing
            if not self.use_gpu:
                os.environ["DETECTOR_BATCH_SIZE"] = os.environ.get("DETECTOR_BATCH_SIZE", "2")
                os.environ["RECOGNITION_BATCH_SIZE"] = os.environ.get("RECOGNITION_BATCH_SIZE", "4")
            # ───────────────────────────────────────────────────────────────

            logger.info("Loading Sur OCRmodels (GPU=%s, Threads=%d, RecBatchSize=%s)...", 
                        self.use_gpu, torch.get_num_threads(), os.environ.get("RECOGNITION_BATCH_SIZE", "32"))
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            self._foundation_predictor = FoundationPredictor()
            self._det_predictor = DetectionPredictor()
            self._rec_predictor = RecognitionPredictor(self._foundation_predictor)
            self._initialized = True
            logger.info("Sur OCRmodels loaded successfully.")
        except Exception as exc:
            logger.error("Failed to initialise Surya OCR: %s", exc)
            raise

    def cleanup(self) -> None:
        self._foundation_predictor = None
        self._det_predictor = None
        self._rec_predictor = None
        self._initialized = False
        logger.info("Sur OCRcleaned up")

    # ── Public extraction ─────────────────────────────────────────────────

    def extract_text(self, image: Image.Image) -> str:
        """Run OCR on ``image`` and return plain text."""
        page = self.extract_structured(image)
        return page.to_plain_text()

    def extract_structured(self, image: Image.Image) -> PageResult:
        """Run Sur OCRand return structured :class:`PageResult`.

        Each detected text line becomes a ``PageElement(type="paragraph")``
        with bbox, confidence, id, reading_order — matching the
        ``pipeline_v2.py`` output format exactly.
        """
        if not self._initialized:
            self.initialize()

        processed = preprocess_for_ocr(image, self.preprocess_options)
        rgb = processed if processed.mode == "RGB" else processed.convert("RGB")

        prediction = self._run_ocr(rgb)
        elements = self._prediction_to_elements(prediction, page_num=1)
        return PageResult(
            page_num=1,
            elements=elements,
            width=rgb.width,
            height=rgb.height,
        )

    # ── Surya inference ───────────────────────────────────────────────────

    def _run_ocr(self, image: Image.Image):
        """Run Surya recognition with detection on a single image."""
        from surya.common.surya.schema import TaskNames

        predictions = self._rec_predictor(
            [image],
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=self._det_predictor,
            sort_lines=True,
            drop_repeated_text=True,
        )
        return predictions[0]

    # ── Result conversion ─────────────────────────────────────────────────

    @staticmethod
    def _prediction_to_elements(prediction, page_num: int = 1) -> List[PageElement]:
        """Convert Surya text lines to ``PageElement`` list.

        Output matches ``pipeline_v2.py``: each block has id, text, bbox,
        confidence, reading_order, cleaned.
        """
        elements: List[PageElement] = []
        reading_order = 0
        for line in prediction.text_lines:
            if not line.text or not line.text.strip():
                continue
            reading_order += 1
            bbox = [
                round(line.bbox[0]),
                round(line.bbox[1]),
                round(line.bbox[2]),
                round(line.bbox[3]),
            ]
            elements.append(
                PageElement(
                    type="paragraph",
                    text=line.text,
                    bbox=bbox,
                    confidence=round(line.confidence, 4),
                    id=f"p{page_num}_b{reading_order}",
                    reading_order=reading_order,
                    cleaned=False,
                )
            )
        return elements

    def get_engine_info(self) -> dict:
        return {
            "name": "SuryaEngine",
            "initialized": self._initialized,
            "lang": self.lang,
        }
