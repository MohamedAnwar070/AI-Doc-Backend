"""OCR engine package.

Public API: :class:`OCREngine` (abstract), :class:`OCRFactory`,
:class:`PaddleEngine` (local), :class:`RemoteOCREngine` (HTTP client),
:class:`GoogleDocAIEngine` (Google Cloud Document AI),
:class:`GoogleVisionAIEngine` (Google Cloud Vision API OCR),
:class:`SuryaEngine` (local Surya OCR).

Engines are imported lazily so the package stays cheap to load even
when only one of them is configured for use.
"""
from __future__ import annotations

from .base import OCREngine
from .factory import OCRFactory

try:
    from .paddle_engine import PaddleEngine  # type: ignore
except ImportError:  # PaddlePaddle unavailable in this environment.
    PaddleEngine = None  # type: ignore[assignment]

try:
    from .remote_engine import RemoteOCREngine  # type: ignore
except ImportError:  # ``httpx`` missing — extremely unlikely in our stack.
    RemoteOCREngine = None  # type: ignore[assignment]

try:
    from .google_engine import GoogleDocAIEngine  # type: ignore
except ImportError:  # ``google-cloud-documentai`` not installed.
    GoogleDocAIEngine = None  # type: ignore[assignment]

try:
    from .google_vision_engine import GoogleVisionAIEngine  # type: ignore
except ImportError:
    GoogleVisionAIEngine = None  # type: ignore[assignment]

try:
    from .surya_engine import SuryaEngine  # type: ignore
except ImportError:  # ``surya-ocr`` not installed.
    SuryaEngine = None  # type: ignore[assignment]

__all__ = [
    "OCREngine",
    "OCRFactory",
    "PaddleEngine",
    "RemoteOCREngine",
    "GoogleDocAIEngine",
    "GoogleVisionAIEngine",
    "SuryaEngine",
]
