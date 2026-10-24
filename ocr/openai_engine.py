"""OpenAI Vision OCR engine — production-grade.

Uses OpenAI's Vision API (GPT-4o or GPT-4o-mini) to extract high-fidelity text content,
structures, layouts, and bounding boxes from page images, and store them perfectly
in the visual editor pipeline.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any, List, Optional

import httpx
from PIL import Image

from core.image_preprocessor import PreprocessOptions, preprocess_for_ocr
from core.layout_engine import PageElement, PageResult
from .base import OCREngine

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT_SECONDS: float = 120.0
_DEFAULT_MAX_RETRIES: int = 3
_PNG_FORMAT: str = "PNG"
_JPEG_QUALITY: int = 90
_BACKOFF_BASE_SECONDS: float = 1.0


class OpenAIOCRError(RuntimeError):
    """Raised when OpenAI API returns an unrecoverable error."""
    pass


class OpenAIEngine(OCREngine):
    """OpenAI Vision OCR adapter — production-grade.

    Uses GPT-4o / GPT-4o-mini to analyze images and return a structured JSON layout,
    then scales normalized coordinate bounding boxes back to absolute image pixels.
    """

    def __init__(
        self,
        *,
        language_hints: str = "en,ta",
        lang: str = "auto",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        preprocess_options: Optional[PreprocessOptions] = None,
        # Accepted for factory-call parity — unused.
        use_gpu: bool = False,
        use_layout: bool = True,
        layout_recovery: bool = True,
        table_as_html: bool = True,
        enable_translation: bool = False,
        translation_target_lang: str = "en",
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.lang = lang or "auto"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.preprocess_options = preprocess_options or PreprocessOptions()
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        if self._initialized:
            return
        from config import settings
        if not settings.openai_api_key:
            raise OpenAIOCRError(
                "OpenAI API Key is missing. Please set OPENAI_API_KEY in your .env file."
            )
        self._initialized = True
        logger.info(
            "OpenAIEngine initialised (model=%s, base_url=%s)",
            settings.openai_ocr_model,
            settings.llm_api_base_openai,
        )

    def cleanup(self) -> None:
        self._initialized = False

    # ── Public extraction ─────────────────────────────────────────────────

    def extract_text(self, image: Image.Image) -> str:
        """Run OCR on ``image`` and return plain text."""
        page = self.extract_structured(image)
        return page.to_plain_text()

    def extract_structured(self, image: Image.Image) -> PageResult:
        """Run OpenAI Vision OCR and return structured :class:`PageResult`."""
        if not self._initialized:
            self.initialize()

        processed = preprocess_for_ocr(image, self.preprocess_options)
        rgb = processed if processed.mode == "RGB" else processed.convert("RGB")

        # Encode image to base64
        base64_image = self._encode_image_to_base64(rgb)

        # Call OpenAI Vision API with retry logic
        response_dict = self._call_vision_api_with_retry(base64_image)

        # Parse response into PageElements
        elements = self._parse_openai_response(response_dict, rgb.width, rgb.height)

        return PageResult(
            page_num=1,
            elements=elements,
            width=rgb.width,
            height=rgb.height,
        )

    # ── API Calls ─────────────────────────────────────────────────────────

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Helper to convert PIL Image to base64 string."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        img_bytes = buf.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def _call_vision_api_with_retry(self, base64_image: str) -> dict[str, Any]:
        """Call the OpenAI Chat Completions endpoint with base64 image payload."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_vision_api_once(base64_image)
            except OpenAIOCRError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI Vision OCR call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        raise OpenAIOCRError(
            f"OpenAI Vision OCR failed after {self.max_retries} attempts: {last_exc}"
        )

    def _call_vision_api_once(self, base64_image: str) -> dict[str, Any]:
        """Perform the actual POST request using httpx."""
        from config import settings

        api_url = f"{settings.llm_api_base_openai.rstrip('/')}/chat/completions"
        model = settings.openai_ocr_model.strip()
        if model.startswith("openai/"):
            model = model.replace("openai/", "", 1)

        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are an expert document parser and OCR system.\n"
            "Analyze the provided document image and extract ALL text content and visual regions into structured layout blocks.\n"
            "Identify elements such as titles, paragraphs, tables, lists, and figures.\n"
            "For each element, you must extract:\n"
            "- type: Must be exactly one of: \"title\", \"paragraph\", \"table\", \"list\", \"figure\"\n"
            "- text: The exact plain text content of the element. For tables, represent the grid content using tabs and newlines.\n"
            "- html: (Mandatory for tables only) A clean HTML <table> representation containing rows <tr> and cells <td>/<th>. Do not include markdown blocks or extra style attributes. Use null for non-table elements.\n"
            "- bbox: [xmin, ymin, xmax, ymax] bounding box as normalized integers from 0 to 1000 representing the element's position on the image. (0,0 is top-left, 1000,1000 is bottom-right).\n"
            "- confidence: A float between 0.0 and 1.0 representing your OCR extraction confidence.\n\n"
            "Strictly follow these rules:\n"
            "1. Extract ALL text on the page. Do not summarize, omit, or truncate anything.\n"
            "2. Maintain reading order (top-to-bottom, left-to-right).\n"
            "3. Preserve bilingual text (e.g. Tamil characters and English) exactly as written.\n"
            "4. Output MUST be a valid JSON object matching the requested schema.\n"
            "5. Do NOT wrap your JSON in Markdown code fences (like ```json ... ```)."
        )

        user_prompt = (
            "Analyze this page image and return the structured layout JSON with 'elements' array."
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {
                "type": "json_object"
            },
            "temperature": 0.1,
        }

        verify: bool | str = True
        try:
            verify = settings.httpx_verify()
        except Exception:
            pass

        with httpx.Client(timeout=self.timeout_seconds, verify=verify) as client:
            response = client.post(api_url, json=payload, headers=headers)

        if response.status_code >= 400:
            raise OpenAIOCRError(
                f"OpenAI Vision OCR API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise OpenAIOCRError(
                f"OpenAI Vision OCR API returned non-JSON response: {response.text[:200]}"
            ) from exc

    # ── Result conversion ─────────────────────────────────────────────────

    def _parse_openai_response(self, response_dict: dict[str, Any], image_width: int, image_height: int) -> List[PageElement]:
        """Convert OpenAI JSON output to structured PageElement list."""
        choices = response_dict.get("choices", [])
        if not choices:
            raise OpenAIOCRError("OpenAI Vision OCR response did not contain choices.")

        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise OpenAIOCRError("OpenAI Vision OCR response message content is empty.")

        # Clean markdown wrappers if present
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse OpenAI Vision OCR JSON payload: %s\nPayload: %s", exc, content)
            raise OpenAIOCRError("OpenAI Vision OCR returned invalid JSON structure.") from exc

        raw_elements = data.get("elements", [])
        if not isinstance(raw_elements, list):
            # Fallback if top-level JSON is directly the elements array or key is different
            if isinstance(data, list):
                raw_elements = data
            else:
                raw_elements = []

        elements: List[PageElement] = []
        reading_order = 0

        for el in raw_elements:
            if not isinstance(el, dict):
                continue

            text = str(el.get("text", "") or "").strip()
            if not text:
                continue

            reading_order += 1
            el_type = str(el.get("type", "paragraph")).lower()
            if el_type not in ("title", "paragraph", "table", "list", "figure"):
                el_type = "paragraph"

            # Scale normalized coordinates [xmin, ymin, xmax, ymax] in [0, 1000] to absolute dimensions
            raw_bbox = el.get("bbox")
            bbox = None
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                try:
                    xmin = float(raw_bbox[0])
                    ymin = float(raw_bbox[1])
                    xmax = float(raw_bbox[2])
                    ymax = float(raw_bbox[3])

                    # Clip to 0-1000 range
                    xmin = max(0.0, min(1000.0, xmin))
                    ymin = max(0.0, min(1000.0, ymin))
                    xmax = max(0.0, min(1000.0, xmax))
                    ymax = max(0.0, min(1000.0, ymax))

                    # Scale to actual pixels
                    x0 = round((xmin / 1000.0) * image_width)
                    y0 = round((ymin / 1000.0) * image_height)
                    x1 = round((xmax / 1000.0) * image_width)
                    y1 = round((ymax / 1000.0) * image_height)

                    bbox = [x0, y0, x1, y1]
                except (ValueError, TypeError) as exc:
                    logger.warning("Failed to parse bounding box %s: %s", raw_bbox, exc)

            # Extract table html if type is table
            html_content = None
            if el_type == "table":
                html_content = el.get("html")
                if not html_content:
                    # Synthesize basic HTML table from text lines as fallback
                    lines = text.splitlines()
                    rows_html = []
                    for line in lines:
                        cells = line.split("\t")
                        cells_html = "".join(f"<td>{c.strip()}</td>" for c in cells if c.strip())
                        if cells_html:
                            rows_html.append(f"<tr>{cells_html}</tr>")
                    if rows_html:
                        html_content = f"<table>{''.join(rows_html)}</table>"

            confidence = 0.95
            if "confidence" in el:
                try:
                    confidence = float(el["confidence"])
                except (ValueError, TypeError):
                    pass

            elements.append(
                PageElement(
                    type=el_type,
                    text=text,
                    html=html_content,
                    bbox=bbox,
                    confidence=round(confidence, 4),
                    id=f"p1_b{reading_order}",
                    reading_order=reading_order,
                    cleaned=True,  # OpenAI results are already pristine / corrected
                )
            )

        # Sort elements by layout coordinates (top to bottom)
        elements.sort(key=lambda e: (e.bbox[1] if e.bbox else 0.0))

        return elements

    def get_engine_info(self) -> dict:
        from config import settings
        return {
            "name": "OpenAIEngine",
            "initialized": self._initialized,
            "model": settings.openai_ocr_model,
        }
