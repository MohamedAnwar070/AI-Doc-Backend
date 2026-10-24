"""OCR engine factory.

Several engines are supported:

``paddle``
    Local PaddleOCR (Tamil + English). Kept for development and as a
    fallback. Requires PaddlePaddle / PaddleOCR to be installed.

``remote``
    HTTP client to an externally-hosted OCR microservice. **Recommended
    for production** — keeps the backend container small and lets the
    OCR service scale (and use GPU) independently.

``google_docai``
    Google Cloud Document AI. Best quality for bilingual scanned PDFs.
    Requires ``google-cloud-documentai`` and a GCP service-account key.

``paddle_vl``
    PaddleOCR-VL-1.5 via HuggingFace transformers. A full Vision Language
    Model (VLM) — best quality for bilingual Tamil/English layout parsing.
    Requires an NVIDIA GPU with ≥8 GB VRAM.

``google_visionai``
    Google Cloud Vision API OCR. Cloud OCR for image-based pages using
    ``DOCUMENT_TEXT_DETECTION``.

``sur_ocr``
    Local Sur OCR(v0.17.x). Strong bilingual Tamil/English detection
    and recognition with per-line bounding boxes. Requires ``surya-ocr``.

The choice is driven by ``settings.ocr_engine`` which in turn reads the
``OCR_ENGINE`` environment variable. The factory is the single place
that knows about engine class names; everything downstream consumes the
:class:`OCREngine` abstract interface.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Union

from .base import OCREngine

logger = logging.getLogger(__name__)

EngineType = Literal["paddle", "paddle_vl", "paddle_vl_ft", "olm_ocr", "remote", "google_docai", "google_visionai", "sur_ocr", "ocr_open_ai", "PaddleOCR1.5"]


# ── Lazy loaders ───────────────────────────────────────────────────────────
# Imports are deferred to avoid pulling Paddle's heavy dependency tree into
# processes that don't need it (e.g. when only the remote engine is in use).

def _lazy_paddle() -> type[OCREngine]:
    from .paddle_engine import PaddleEngine
    return PaddleEngine


def _lazy_remote() -> type[OCREngine]:
    from .remote_engine import RemoteOCREngine
    return RemoteOCREngine


def _lazy_google_docai() -> type[OCREngine]:
    from .google_engine import GoogleDocAIEngine
    return GoogleDocAIEngine


def _lazy_google_visionai() -> type[OCREngine]:
    from .google_vision_engine import GoogleVisionAIEngine
    return GoogleVisionAIEngine


def _lazy_paddle_vl() -> type[OCREngine]:
    from .paddle_vl_engine import PaddleVLEngine
    return PaddleVLEngine


def _lazy_paddle_vl_remote() -> type[OCREngine]:
    from .paddle_vl_remote_engine import PaddleVLRemoteEngine
    return PaddleVLRemoteEngine


def _lazy_paddle_vl_or_remote() -> type[OCREngine]:
    """Return the remote engine when ``PADDLE_VL_API_URL`` is configured,
    otherwise fall back to the local VLM engine."""
    from config import settings
    if settings.paddle_vl_api_url:
        logger.info(
            "PaddleOCR1.5: using remote API at %s", settings.paddle_vl_api_url
        )
        return _lazy_paddle_vl_remote()
    logger.info("PaddleOCR1.5: no PADDLE_VL_API_URL set, using local VLM engine")
    return _lazy_paddle_vl()


def _lazy_olm_ocr() -> type[OCREngine]:
    from .olmocr_engine import OlmOcrEngine
    return OlmOcrEngine


def _lazy_sur_ocr() -> type[OCREngine]:
    from .surya_engine import SuryaEngine
    return SuryaEngine


def _lazy_ocr_open_ai() -> type[OCREngine]:
    from .openai_engine import OpenAIEngine
    return OpenAIEngine


def _lazy_paddle_vl_ft() -> type[OCREngine]:
    from .paddle_vl_finetuned_engine import PaddleVLFinetunedEngine
    return PaddleVLFinetunedEngine


_ENGINE_LOADERS: dict[str, Callable[[], type[OCREngine]]] = {
    "paddle": _lazy_paddle,
    "paddle_vl": _lazy_paddle_vl,
    "PaddleOCR1.5": _lazy_paddle_vl_or_remote,
    "paddle_vl_ft": _lazy_paddle_vl_ft,
    "olm_ocr": _lazy_olm_ocr,
    "remote": _lazy_remote,
    "google_docai": _lazy_google_docai,
    "google_visionai": _lazy_google_visionai,
    "sur_ocr": _lazy_sur_ocr,
    "ocr_open_ai": _lazy_ocr_open_ai,
}


class OCRFactory:
    """Static factory for :class:`OCREngine` implementations."""

    @classmethod
    def get_engine(cls, engine_type: EngineType, **kwargs: Any) -> OCREngine:
        """Return a fresh OCR engine instance.

        Parameters
        ----------
        engine_type:
            ``"paddle"``, ``"paddle_vl"``, ``"remote"``, ``"google_docai"``, or ``"google_visionai"``.
        **kwargs:
            Forwarded to the engine constructor. Unknown kwargs are
            ignored by the underlying engines (they accept ``**_``).

        Raises
        ------
        ValueError
            If ``engine_type`` is not a known engine.
        """
        if engine_type not in _ENGINE_LOADERS:
            valid = ", ".join(sorted(_ENGINE_LOADERS))
            raise ValueError(
                f"Unsupported OCR engine: '{engine_type}'. Valid options: {valid}."
            )

        logger.debug("Creating OCR engine: %s", engine_type)
        engine_class = _ENGINE_LOADERS[engine_type]()
        return engine_class(**kwargs)

    @classmethod
    def list_engines(cls) -> list[str]:
        """Return the registered engine identifiers."""
        return list(_ENGINE_LOADERS.keys())
