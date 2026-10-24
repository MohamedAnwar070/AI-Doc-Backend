"""PaddleOCR-VL-1.5 Remote Engine — HTTP client to a deployed ``paddle_vl_api.py`` service.

When ``PADDLE_VL_API_URL`` is configured, ``OCR_ENGINE=PaddleOCR1.5`` delegates
all VLM inference to the remote microservice instead of loading the 16 GB model
in-process.  This keeps the main backend container small and CPU-only while the
heavy VLM runs on a separate node (GPU or high-CPU).

Wire format (matches :mod:`paddle_vl_api`)::

    POST {base_url}/ocr
    Content-Type: multipart/form-data

    image    : <PNG bytes>
    language : "auto"
    layout   : "true"
    model    : "paddle_vl"

    Response:
    {
      "engine":   "paddle_vl",
      "language": "auto",
      "elements": [
        {
          "type":       "title" | "paragraph" | "table" | ...,
          "text":       "...",
          "html":       "..." | null,
          "bbox":       [x0, y0, x1, y1] | null,
          "confidence": 0.95,
          "language":   "ta" | "en" | "mixed" | null
        }
      ]
    }

The engine converts the ``elements`` array into :class:`LayoutBlock` /
:class:`LayoutPage` objects so the existing worker pipeline (sequential
execution, Tamil normalisation, page image saving) works unchanged.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import List, Optional

from PIL import Image

from .base import OCREngine
from .layout_schema import (
    BlockType,
    LanguageCode,
    LayoutBlock,
    LayoutPage,
    LayoutTable,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_DEFAULT_TIMEOUT_SECONDS: float = 300.0
_RETRY_ATTEMPTS: int = 2
_MAX_IMAGE_BYTES: int = 25 * 1024 * 1024  # 25 MiB safety cap
_JPEG_QUALITY: int = 92

# Type mapping: paddle_vl_api.py sends "title" for headings (line 158),
# but our LayoutBlock schema uses "heading".
_API_TYPE_TO_BLOCK_TYPE: dict[str, BlockType] = {
    "title": "heading",
    "heading": "heading",
    "paragraph": "paragraph",
    "table": "table",
    "list": "list",
    "list_item": "list_item",
    "figure": "figure",
    "header": "header",
    "footer": "footer",
}


class PaddleVLRemoteEngineError(RuntimeError):
    """Raised when the remote PaddleOCR-VL API returns an unrecoverable error."""


class PaddleVLRemoteEngine(OCREngine):
    """Client-side engine that delegates VLM inference to a deployed paddle_vl_api service.

    Implements the same ``extract_page()`` → :class:`LayoutPage` interface as
    :class:`PaddleVLEngine`, so the worker pipeline treats it identically.

    Parameters
    ----------
    base_url:
        Base URL of the deployed paddle_vl_api (e.g. ``https://paddle-vl.example.com``).
    api_key:
        Bearer token for the remote API's ``OCR_API_KEY`` authentication.
    timeout_seconds:
        HTTP timeout per request.  VLM inference on CPU can take 30–60 s per page.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        api_key: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        # Accept and ignore kwargs forwarded by the factory / job_manager
        # (use_gpu, load_in_8bit, etc.) — they are for the local engine.
        **_ignored_kwargs,
    ) -> None:
        super().__init__()
        if not base_url:
            raise ValueError(
                "PaddleVLRemoteEngine requires PADDLE_VL_API_URL to be set. "
                "Either set the env var or use OCR_ENGINE=paddle_vl for local mode."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self._initialized: bool = False

    # ── OCREngine lifecycle ──────────────────────────────────────────────

    def initialize(self) -> None:
        """Probe the remote ``/health`` endpoint to fail fast on misconfiguration."""
        if self._initialized:
            return
        try:
            self._probe_health()
            self._initialized = True
            logger.info(
                "PaddleVLRemoteEngine initialised (url=%s)", self.base_url
            )
        except Exception as exc:
            logger.warning(
                "PaddleVLRemoteEngine health-check failed (proceeding anyway): %s", exc
            )
            self._initialized = True

    def cleanup(self) -> None:
        """No local resources to release."""
        self._initialized = False

    def extract_text(self, image: Image.Image) -> str:
        """Run OCR via the remote API and return plain text."""
        page = self.extract_page(image, page_number=1)
        return "\n\n".join(blk.text for blk in page.blocks if blk.text)

    # ── Structured extraction (the path the VLM worker uses) ─────────────

    def extract_page(self, image: Image.Image, page_number: int = 1) -> LayoutPage:
        """Send image to the remote API, parse response into a :class:`LayoutPage`.

        This method signature matches :meth:`PaddleVLEngine.extract_page` so the
        worker's VLM path (``process_single_page`` Path 3) works unchanged.
        """
        if not self._initialized:
            self.initialize()

        page_w, page_h = image.size
        payload = self._call_with_retry(image)

        elements = payload.get("elements", [])
        blocks: List[LayoutBlock] = []
        tables: List[LayoutTable] = []

        for idx, elem in enumerate(elements):
            if not isinstance(elem, dict):
                continue

            # Map API type to LayoutBlock type
            raw_type = str(elem.get("type", "paragraph")).lower()
            block_type: BlockType = _API_TYPE_TO_BLOCK_TYPE.get(raw_type, "paragraph")

            text = str(elem.get("text", "") or "")
            if not text and block_type != "figure":
                continue

            # Parse bbox [x0, y0, x1, y1] → x, y, width, height
            bbox = elem.get("bbox")
            x, y, width, height = 0.0, 0.0, 0.0, 0.0
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y = float(bbox[0]), float(bbox[1])
                width = float(bbox[2]) - x
                height = float(bbox[3]) - y

            # Language code
            lang_raw = elem.get("language") or "unknown"
            language: LanguageCode = (
                lang_raw if lang_raw in ("ta", "en", "mixed", "unknown") else "unknown"
            )

            block_id = f"p{page_number}_b{idx:03d}"
            blocks.append(
                LayoutBlock(
                    block_id=block_id,
                    type=block_type,
                    text=text,
                    html=elem.get("html"),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=float(elem.get("confidence") or 0.0),
                    language=language,
                    reading_order=idx + 1,
                )
            )

        return LayoutPage(
            page_number=page_number,
            width_px=page_w,
            height_px=page_h,
            dpi=0,
            blocks=blocks,
            tables=tables,
        )

    # ── HTTP plumbing ────────────────────────────────────────────────────

    def _call_with_retry(self, image: Image.Image) -> dict:
        """Call the remote OCR API with a retry loop for transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                return self._call_once(image)
            except PaddleVLRemoteEngineError:
                raise  # non-transient (4xx, bad JSON)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Remote PaddleVL call failed (attempt %d/%d): %s",
                    attempt + 1,
                    _RETRY_ATTEMPTS,
                    exc,
                )
        raise PaddleVLRemoteEngineError(
            f"Remote PaddleVL call failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        )

    def _call_once(self, image: Image.Image) -> dict:
        import httpx

        url = f"{self.base_url}/ocr"
        image_bytes = self._encode_image(image)

        files = {"image": ("page.png", image_bytes, "image/png")}
        data = {
            "language": "auto",
            "layout": "true",
            "model": "paddle_vl",
        }
        headers = self._build_auth_headers()

        logger.info(
            "Remote PaddleVL OCR request started: url=%s image=%dx%d bytes=%d timeout=%.0fs",
            url,
            image.width,
            image.height,
            len(image_bytes),
            self.timeout_seconds,
        )

        t0 = time.perf_counter()
        try:
            timeout = httpx.Timeout(self.timeout_seconds, connect=30.0)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, files=files, data=data, headers=headers)
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "Remote PaddleVL OCR request timed out after %d ms "
                "(configured timeout=%.0fs)",
                elapsed_ms,
                self.timeout_seconds,
            )
            raise
        except httpx.HTTPError:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception(
                "Remote PaddleVL OCR request failed after %d ms",
                elapsed_ms,
            )
            raise

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if response.status_code >= 400:
            raise PaddleVLRemoteEngineError(
                f"Remote PaddleVL API returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise PaddleVLRemoteEngineError(
                f"Remote PaddleVL API returned non-JSON body: {response.text[:200]}"
            ) from exc

        elem_count = len(result.get("elements", []))
        logger.info(
            "Remote PaddleVL OCR response received: status=%d elements=%d elapsed_ms=%d",
            response.status_code,
            elem_count,
            elapsed_ms,
        )
        return result

    def _probe_health(self) -> None:
        import httpx

        url = f"{self.base_url}/health"
        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, headers=self._build_auth_headers())
        if response.status_code >= 400:
            raise PaddleVLRemoteEngineError(
                f"Health check failed: HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        body = response.json()
        logger.info(
            "PaddleVL API health: status=%s, model=%s, gpu=%s",
            body.get("status"),
            body.get("model"),
            body.get("gpu_enabled"),
        )

    def _build_auth_headers(self) -> dict[str, str]:
        """Construct ``Authorization`` header from the API key (Bearer token)."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    # ── Encoding helpers ─────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image: Image.Image) -> bytes:
        """Encode ``image`` as PNG bytes, falling back to JPEG if too large."""
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        if len(data) <= _MAX_IMAGE_BYTES:
            return data

        logger.debug(
            "Image PNG (%d bytes) exceeds cap; falling back to JPEG",
            len(data),
        )
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return buffer.getvalue()

    def get_engine_info(self) -> dict:
        return {
            "name": "PaddleVLRemoteEngine",
            "base_url": self.base_url,
            "initialized": self._initialized,
        }
