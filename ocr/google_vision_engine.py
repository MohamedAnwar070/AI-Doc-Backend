from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

import google.auth
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from PIL import Image

from core.image_preprocessor import PreprocessOptions, preprocess_for_ocr
from core.layout_engine import PageElement, PageResult
from .base import OCREngine

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_MAX_RETRIES = 3
_PNG_SIZE_SOFT_LIMIT_BYTES = 10 * 1024 * 1024
_JPEG_QUALITY = 90
_BACKOFF_BASE_SECONDS = 1.0


class GoogleVisionAIError(RuntimeError):
    pass


class GoogleVisionAIEngine(OCREngine):
    def __init__(
        self,
        *,
        credentials_path: str = "",
        language_hints: str = "en,ta",
        lang: str = "auto",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        preprocess_options: Optional[PreprocessOptions] = None,
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
        self.lang = lang or "auto"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.preprocess_options = (
            preprocess_options or PreprocessOptions()
        ).for_language(self.lang if self.lang != "auto" else "en")
        self.use_gpu = use_gpu
        self.use_layout = use_layout
        self.layout_recovery = layout_recovery
        self.table_as_html = table_as_html
        self.enable_translation = enable_translation
        self.translation_target_lang = translation_target_lang
        self._language_hints = self._resolve_language_hints(language_hints, self.lang)
        self._credentials: Any = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
                    self.credentials_path
                )
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(GoogleAuthRequest())
            self._credentials = credentials
            self._initialized = True
            logger.info(
                "GoogleVisionAIEngine initialised (lang=%s, hints=%s)",
                self.lang,
                self._language_hints,
            )
        except Exception as exc:
            logger.error("GoogleVisionAIEngine init failed: %s", exc)
            raise GoogleVisionAIError(
                "Google Vision AI initialisation failed. Ensure application default "
                "credentials are available or set GOOGLE_VISIONAI_CREDENTIALS."
            ) from exc

    def cleanup(self) -> None:
        self._credentials = None
        self._initialized = False

    def extract_text(self, image: Image.Image) -> str:
        page = self.extract_structured(image)
        return page.to_plain_text()

    def extract_structured(self, image: Image.Image) -> PageResult:
        if not self._initialized:
            self.initialize()
        prepared = preprocess_for_ocr(image, self.preprocess_options)
        payload = self._call_with_retry(self._encode_image(prepared))
        return self._parse_response(payload)

    def _call_with_retry(self, image_bytes: bytes) -> dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_once(image_bytes)
            except GoogleVisionAIError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Google Vision OCR call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        raise GoogleVisionAIError(
            f"Google Vision OCR failed after {self.max_retries} attempts: {last_exc}"
        )

    def _call_once(self, image_bytes: bytes) -> dict[str, Any]:
        token = self._get_access_token()
        request_payload: dict[str, Any] = {
            "features": [
                {
                    "type": "DOCUMENT_TEXT_DETECTION",
                    "model": "builtin/latest",
                    "maxResults": 50,
                }
            ],
            "image": {"content": base64.b64encode(image_bytes).decode("utf-8")},
        }
        if self._language_hints:
            request_payload["imageContext"] = {"languageHints": self._language_hints}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                "https://vision.googleapis.com/v1/images:annotate",
                headers=headers,
                json={"requests": [request_payload]},
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise RuntimeError(
                f"Google Vision OCR transient HTTP {response.status_code}: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise GoogleVisionAIError(
                f"Google Vision OCR returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise GoogleVisionAIError(
                f"Google Vision OCR returned non-JSON body: {response.text[:200]}"
            ) from exc

    def _get_access_token(self) -> str:
        if not self._credentials:
            self.initialize()
        if not self._credentials.valid:
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

    @staticmethod
    def _encode_image(image: Image.Image) -> bytes:
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG", optimize=True)
        png_bytes = buffer.getvalue()
        if len(png_bytes) <= _PNG_SIZE_SOFT_LIMIT_BYTES:
            return png_bytes
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return buffer.getvalue()

    def _parse_response(self, payload: dict[str, Any]) -> PageResult:
        responses = payload.get("responses", [])
        if not responses:
            raise GoogleVisionAIError("Google Vision OCR response did not contain responses")
        first = responses[0]
        if "error" in first:
            message = first["error"].get("message", "Unknown Vision API error")
            raise GoogleVisionAIError(f"Vision API Error: {message}")
        full_text = first.get("fullTextAnnotation", {})
        elements = self._parse_elements(full_text)
        if not elements:
            text = str(full_text.get("text", "") or "").strip()
            if text:
                elements = [PageElement(type="paragraph", text=text, confidence=0.0)]
        return PageResult(page_num=0, elements=elements)

    @classmethod
    def _parse_elements(cls, full_text_annotation: dict[str, Any]) -> list[PageElement]:
        elements: list[PageElement] = []
        for page in full_text_annotation.get("pages", []):
            for block in page.get("blocks", []):
                block_property = block.get("property", {}) or {}
                for paragraph in block.get("paragraphs", []):
                    text = cls._paragraph_text(paragraph).strip()
                    if not text:
                        continue
                    paragraph_property = paragraph.get("property", {}) or {}
                    elements.append(
                        PageElement(
                            type="paragraph",
                            text=text,
                            bbox=cls._bounding_poly_to_bbox(
                                paragraph.get("boundingBox", {}) or {}
                            ),
                            language=cls._detect_language(paragraph_property, block_property),
                            confidence=float(
                                paragraph.get("confidence")
                                or block.get("confidence")
                                or 0.0
                            ),
                        )
                    )
        elements.sort(
            key=lambda element: (
                element.bbox[1] if element.bbox else 0.0,
                element.bbox[0] if element.bbox else 0.0,
            )
        )
        return elements

    @staticmethod
    def _paragraph_text(paragraph: dict[str, Any]) -> str:
        parts: list[str] = []
        for word in paragraph.get("words", []):
            symbols = word.get("symbols", []) or []
            if not symbols:
                continue
            parts.append("".join(str(symbol.get("text", "")) for symbol in symbols))
            break_type = (
                symbols[-1]
                .get("property", {})
                .get("detectedBreak", {})
                .get("type", "")
            )
            if break_type in {"SPACE", "SURE_SPACE"}:
                parts.append(" ")
            elif break_type in {"EOL_SURE_SPACE", "LINE_BREAK"}:
                parts.append("\n")
            elif break_type == "HYPHEN":
                parts.append("-")
        return "".join(parts)

    @staticmethod
    def _bounding_poly_to_bbox(bounding_box: dict[str, Any]) -> list[float] | None:
        vertices = bounding_box.get("vertices", []) or []
        coords = [
            (float(vertex.get("x", 0.0) or 0.0), float(vertex.get("y", 0.0) or 0.0))
            for vertex in vertices
        ]
        if not coords:
            return None
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _detect_language(*properties: dict[str, Any]) -> str | None:
        for prop in properties:
            detected = prop.get("detectedLanguages", []) or []
            for item in detected:
                code = str(item.get("languageCode", "") or "").strip()
                if code:
                    return code
        return None

    @staticmethod
    def _resolve_language_hints(hints_csv: str, lang: str) -> list[str]:
        if hints_csv.strip():
            return [item.strip() for item in hints_csv.split(",") if item.strip()]
        normalized_lang = (lang or "").strip().lower()
        if normalized_lang and normalized_lang != "auto":
            return [normalized_lang]
        return []
