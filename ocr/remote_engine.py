"""Remote OCR engine — HTTP client to an externally-hosted OCR microservice.

This is the **production** engine. The main backend container ships
*without* OCR model weights; instead it POSTs page images to a separately
hosted OCR API and receives structured JSON back. That keeps the backend
image small, lets the OCR service scale (and use GPU) independently, and
lets you swap the underlying engine (Surya / Tesseract / PaddleOCR) by
re-deploying only the OCR service.

Wire format
-----------

Request (``POST {OCR_REMOTE_URL}/ocr``)::

    Content-Type: multipart/form-data

    image     : <PNG/JPEG bytes>
    language  : "ta" | "en" | "auto"          # optional, default "auto"
    layout    : "true" | "false"               # optional, default "true"
    model     : "<model id>"                   # optional, default OCR_REMOTE_MODEL

Response::

    {
      "engine":   "surya",
      "language": "ta",
      "elements": [
        {
          "type": "paragraph" | "title" | "table" | "list" | "figure",
          "text": "...",
          "html": "...",            # tables only
          "bbox": [x0, y0, x1, y1], # optional
          "confidence": 0.97        # optional
        },
        ...
      ]
    }

Authentication: either HTTP Basic (``OCR_REMOTE_AUTH=user:pass``) or a
bearer token (``OCR_REMOTE_API_KEY=...``). Both are optional.

Preprocessing
-------------
The engine **always** runs the standard
:func:`core.image_preprocessor.preprocess_for_ocr` pipeline on the image
before uploading it. This makes Tamil OCR quality independent of the
remote engine implementation: deskew, denoise, contrast, and CLAHE are
applied identically whether the heavy model is Surya, Tesseract or
PaddleOCR.

The engine never blocks on the FastAPI event loop; it uses ``httpx`` in
synchronous mode here because OCR runs inside a worker process.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

from PIL import Image

from core.image_preprocessor import PreprocessOptions, preprocess_for_ocr
from core.layout_engine import PageElement, PageResult
from .base import OCREngine

logger = logging.getLogger(__name__)


# ── Constants (no magic numbers in code) ───────────────────────────────────

_DEFAULT_TIMEOUT_SECONDS: float = 120.0
_RETRY_ATTEMPTS: int = 2
_PNG_FORMAT: str = "PNG"
_JPEG_QUALITY: int = 92
_MAX_IMAGE_BYTES: int = 25 * 1024 * 1024  # 25 MiB safety cap


class RemoteOCREngineError(RuntimeError):
    """Raised when the remote OCR service returns an unrecoverable error."""


class RemoteOCREngine(OCREngine):
    """Client-side OCR engine that delegates work to an external HTTP service."""

    def __init__(
        self,
        base_url: str,
        *,
        lang: str = "auto",
        use_layout: bool = True,
        model: str = "surya",
        auth: str = "",
        api_key: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        verify_tls: bool | str = True,
        preprocess_options: Optional[PreprocessOptions] = None,
        # Kept for signature parity with PaddleEngine — unused remotely.
        use_gpu: bool = False,
        layout_recovery: bool = True,
        table_as_html: bool = True,
        enable_translation: bool = False,
        translation_target_lang: str = "en",
    ) -> None:
        super().__init__()
        if not base_url:
            raise ValueError("RemoteOCREngine requires OCR_REMOTE_URL to be set")

        self.base_url = base_url.rstrip("/")
        self.lang = lang or "auto"
        self.use_layout = use_layout
        self.model = model or "surya"
        self.auth = auth
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.verify_tls = verify_tls
        self.preprocess_options = (
            preprocess_options or PreprocessOptions()
        ).for_language(self.lang)

        # Unused locally — we accept them for factory-call uniformity.
        self.use_gpu = use_gpu
        self.layout_recovery = layout_recovery
        self.table_as_html = table_as_html
        self.enable_translation = enable_translation
        self.translation_target_lang = translation_target_lang

        self._initialized = False

    # ── OCREngine ABC implementation ──────────────────────────────────────

    def initialize(self) -> None:
        """Probe the remote ``/health`` endpoint once so failures show up early."""
        if self._initialized:
            return
        try:
            self._probe_health()
            self._initialized = True
            logger.info(
                "RemoteOCREngine initialised (url=%s, model=%s, lang=%s)",
                self.base_url,
                self.model,
                self.lang,
            )
        except Exception as exc:
            logger.error("RemoteOCREngine health-check failed: %s", exc)
            raise

    def cleanup(self) -> None:
        """Nothing to release locally — server-side resources are managed remotely."""
        self._initialized = False

    def extract_text(self, image: Image.Image) -> str:
        """Run OCR on ``image`` and return a single flattened text string."""
        page = self.extract_structured(image)
        return page.to_plain_text()

    # ── Structured extraction (the path the worker uses) ──────────────────

    def extract_structured(self, image: Image.Image) -> PageResult:
        """Call the remote service and translate the JSON response into a :class:`PageResult`."""
        if not self._initialized:
            self.initialize()

        payload = self._call_with_retry(image)
        elements = self._parse_elements(payload.get("elements", []))
        return PageResult(page_num=0, elements=elements)

    # ── HTTP plumbing ─────────────────────────────────────────────────────

    def _call_with_retry(self, image: Image.Image) -> dict[str, Any]:
        """Call the remote OCR service with a small retry loop for transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                return self._call_once(image)
            except RemoteOCREngineError:
                raise  # non-transient
            except Exception as exc:  # noqa: BLE001 — retry envelope
                last_exc = exc
                logger.warning(
                    "Remote OCR call failed (attempt %d/%d): %s",
                    attempt + 1,
                    _RETRY_ATTEMPTS,
                    exc,
                )
        raise RemoteOCREngineError(
            f"Remote OCR call failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        )

    def _call_once(self, image: Image.Image) -> dict[str, Any]:
        import httpx  # local import — keeps cold-start fast

        url = f"{self.base_url}/ocr"
        # Apply the shared preprocessing pipeline so OCR quality (deskew,
        # denoise, CLAHE, Tamil stroke preservation) does not depend on
        # whatever engine the remote service happens to run.
        prepared = preprocess_for_ocr(image, self.preprocess_options)
        image_bytes = self._encode_image(prepared)
        files = {"image": ("page.png", image_bytes, "image/png")}
        data = {
            "language": self.lang,
            "layout": "true" if self.use_layout else "false",
            "model": self.model,
        }
        headers = self._build_auth_headers()

        with httpx.Client(timeout=self.timeout_seconds, verify=self.verify_tls) as client:
            response = client.post(url, files=files, data=data, headers=headers)

        if response.status_code >= 400:
            raise RemoteOCREngineError(
                f"Remote OCR returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RemoteOCREngineError(
                f"Remote OCR returned non-JSON body: {response.text[:200]}"
            ) from exc

    def _probe_health(self) -> None:
        import httpx

        url = f"{self.base_url}/health"
        with httpx.Client(timeout=10.0, verify=self.verify_tls) as client:
            response = client.get(url, headers=self._build_auth_headers())
        if response.status_code >= 400:
            raise RemoteOCREngineError(
                f"Health check failed: HTTP {response.status_code}: {response.text[:200]}"
            )

    def _build_auth_headers(self) -> dict[str, str]:
        """Construct ``Authorization`` header from either Basic creds or bearer token."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        if self.auth:
            encoded = base64.b64encode(self.auth.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        return {}

    # ── Encoding / parsing helpers ────────────────────────────────────────

    @staticmethod
    def _encode_image(image: Image.Image) -> bytes:
        """Encode ``image`` as PNG bytes, falling back to JPEG if too large."""
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format=_PNG_FORMAT, optimize=True)
        data = buffer.getvalue()
        if len(data) <= _MAX_IMAGE_BYTES:
            return data

        logger.debug(
            "Image PNG (%d bytes) exceeds remote OCR cap; falling back to JPEG",
            len(data),
        )
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return buffer.getvalue()

    @staticmethod
    def _parse_elements(raw_elements: list[Any]) -> list[PageElement]:
        """Translate the wire-format ``elements`` array into :class:`PageElement` objects."""
        out: list[PageElement] = []
        for raw in raw_elements:
            if not isinstance(raw, dict):
                continue
            element_type = str(raw.get("type", "paragraph")).lower()
            text = str(raw.get("text", "") or "")
            if not text and element_type != "figure":
                continue
            out.append(
                PageElement(
                    type=element_type,
                    text=text,
                    html=raw.get("html"),
                    bbox=raw.get("bbox"),
                    language=raw.get("language"),
                    confidence=float(raw.get("confidence") or 0.0),
                )
            )
        return out
