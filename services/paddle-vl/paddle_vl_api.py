"""
Standalone Local PaddleOCR-VL-1.5 API Server.

Exposes a ``/ocr`` endpoint matching the wire format expected by the
backend's :class:`ocr.paddle_vl_remote_engine.PaddleVLRemoteEngine` client.

Running this service locally for development::

    python paddle_vl_api.py --host 0.0.0.0 --port 8001

Or via the project's Kubernetes manifests::

    kubectl apply -f k8s/paddle-vl/

Why this file exists
--------------------
The main backend container ships *without* the ~16 GB PaddleOCR-VL model.
Instead the backend POSTs page images to this microservice (deployed in
the ``llm-llamacpp`` namespace, exposed at ``https://paddle-vl.example.com``)
and receives structured JSON back. This keeps the backend image small,
lets the OCR service scale (and use GPU) independently, and lets us swap
the underlying engine by redeploying only this service.

Improvements over the previous version
--------------------------------------
* **Bounded request-deduplication cache** with TTL — identical image
  bytes + form parameters within ``PADDLE_VL_DEDUP_TTL_SECONDS`` share a
  single VLM inference. Retries from the backend no longer pay the full
  inference cost twice.
* **Proper inflight task tracking** — every running task is registered
  in a name-based set; on shutdown we await all of them up to a deadline
  so VLM workers aren't killed mid-inference.
* **Per-request timeout** (``PADDLE_VL_REQUEST_TIMEOUT``) — a stuck
  inference never pins the event loop forever.
* **Hash uses ``xxhash``** when available, else ``hashlib.blake2b`` —
  SHA-256 of multi-megabyte page images is unnecessary on the hot path.
* **Result cache** (``PADDLE_VL_RESULT_CACHE_SIZE`` entries,
  ``PADDLE_VL_RESULT_CACHE_TTL`` seconds) — quick retries within the TTL
  return the cached payload without re-running the model.
* **Health endpoint** distinguishes ``healthy`` / ``loading`` /
  ``degraded`` / ``error`` so k8s readiness probes don't route traffic
  before the model is warm.
* **Strict structured-output preservation** — official PaddleOCR-VL
  ``parsing_res_list`` regions are kept as blocks; Markdown fallback
  only kicks in when the official pipeline returns no structured data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

# ── Project imports ────────────────────────────────────────────────────────
# These run AFTER ``sys.path`` is bootstrapped (the k8s pod has /app as
# cwd which contains the ``ocr/`` package). When launched as a script
# via ``python paddle_vl_api.py`` from the project root this also works.
try:
    from ocr.layout_schema import LayoutBlock, LayoutPage
    from ocr.paddle_vl_engine import (
        _detect_language,
        _is_markdown_table,
        _markdown_table_to_html,
        _markdown_to_layout_table,
        _split_paragraphs,
    )
except ImportError:  # pragma: no cover — script-mode fallback
    # Allow ``python paddle_vl_api.py`` from any cwd by injecting the
    # project root onto sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ocr.layout_schema import LayoutBlock, LayoutPage  # type: ignore[no-redef]
    from ocr.paddle_vl_engine import (  # type: ignore[no-redef]
        _detect_language,
        _is_markdown_table,
        _markdown_table_to_html,
        _markdown_to_layout_table,
        _split_paragraphs,
    )

# ── Optional xxhash ────────────────────────────────────────────────────────
# xxhash is ~10x faster than SHA-256 for non-cryptographic use. It's
# not in the standard library and not strictly required (we fall back
# to blake2b which ships with Python 3.6+).
try:  # pragma: no cover — import-only branch
    import xxhash  # type: ignore[import-untyped]

    def _fast_digest(data: bytes) -> str:
        return xxhash.xxh64(data).hexdigest()

except ImportError:  # pragma: no cover — fallback when xxhash missing

    def _fast_digest(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=16).hexdigest()


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("paddle_vl_api")


# ── Helpers used in module-level env config ────────────────────────────────
def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── Environment-driven configuration ───────────────────────────────────────
API_KEY = os.environ.get("OCR_API_KEY") or None
LOAD_IN_8BIT = os.environ.get("PADDLE_VL_LOAD_IN_8BIT", "false").lower() in {"1", "true", "yes", "on"}
LOAD_IN_4BIT = os.environ.get("PADDLE_VL_LOAD_IN_4BIT", "false").lower() in {"1", "true", "yes", "on"}
MAX_NEW_TOKENS = int(os.environ.get("PADDLE_VL_MAX_NEW_TOKENS", "4096"))
CACHE_DIR = os.environ.get("PADDLE_VL_CACHE_DIR") or None
USE_GPU = os.environ.get("GPU_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
PADDLE_OCR_DEVICE = os.environ.get("PADDLE_OCR_DEVICE") or ("gpu" if USE_GPU else "cpu")

# Inference engine selection. ``None`` uses PaddleOCR's default
# (PaddlePaddle local). ``"transformers"`` is needed when 8-bit / 4-bit
# quantisation is requested (bitsandbytes is only available via the
# transformers backend). Future-proofs GPU deployments that want to
# quantise the VLM to fit larger models on smaller VRAM.
PIPELINE_ENGINE: Optional[str] = os.environ.get("PADDLE_VL_ENGINE") or None

# Optional VLM inference service (vllm-server / sglang-server /
# fastdeploy-server). When set, the pipeline offloads the VLM stage to
# that service while layout analysis still runs locally. Empty = local.
VL_REC_BACKEND: Optional[str] = os.environ.get("PADDLE_VL_REC_BACKEND") or None
VL_REC_SERVER_URL: Optional[str] = os.environ.get("PADDLE_VL_REC_SERVER_URL") or None
VL_REC_MAX_CONCURRENCY: Optional[int] = (
    int(os.environ["PADDLE_VL_REC_MAX_CONCURRENCY"])
    if os.environ.get("PADDLE_VL_REC_MAX_CONCURRENCY")
    else None
)
VL_REC_API_MODEL_NAME: Optional[str] = (
    os.environ.get("PADDLE_VL_REC_API_MODEL_NAME") or None
)
VL_REC_API_KEY: Optional[str] = os.environ.get("PADDLE_VL_REC_API_KEY") or None

# Sampling defaults — low temperature for deterministic OCR output.
TEMPERATURE: Optional[float] = (
    float(os.environ["PADDLE_VL_TEMPERATURE"])
    if os.environ.get("PADDLE_VL_TEMPERATURE")
    else None
)
TOP_P: Optional[float] = (
    float(os.environ["PADDLE_VL_TOP_P"])
    if os.environ.get("PADDLE_VL_TOP_P")
    else None
)
REPETITION_PENALTY: Optional[float] = (
    float(os.environ["PADDLE_VL_REPETITION_PENALTY"])
    if os.environ.get("PADDLE_VL_REPETITION_PENALTY")
    else None
)

# VLM image preprocessing bounds. None → use PaddleOCR defaults
# (clamped to its internal min/max). Setting these prevents OOM on
# dense pages and keeps small receipts at readable resolution.
MIN_PIXELS: Optional[int] = (
    int(os.environ["PADDLE_VL_MIN_PIXELS"])
    if os.environ.get("PADDLE_VL_MIN_PIXELS")
    else None
)
MAX_PIXELS: Optional[int] = (
    int(os.environ["PADDLE_VL_MAX_PIXELS"])
    if os.environ.get("PADDLE_VL_MAX_PIXELS")
    else None
)

# Layout shape mode for v1.5: ``rect`` | ``quad`` | ``poly`` | ``auto``.
# v1.5 supports irregular-shaped bounding boxes; ``auto`` lets the
# pipeline pick the right shape per region. Defaults to empty (opt-in)
# so the kwarg is only forwarded to PaddleOCRVL when the user
# explicitly sets the env var — older paddleocr builds reject unknown
# kwargs at construction time.
LAYOUT_SHAPE_MODE: str = os.environ.get("PADDLE_VL_LAYOUT_SHAPE_MODE", "")
PROMPT_LABEL: Optional[str] = os.environ.get("PADDLE_VL_PROMPT_LABEL") or None

# v1.5 new feature toggles. All default to ``None`` (use framework default).
USE_CHART_RECOGNITION = _env_bool("PADDLE_VL_USE_CHART_RECOGNITION")
USE_SEAL_RECOGNITION = _env_bool("PADDLE_VL_USE_SEAL_RECOGNITION")
USE_OCR_FOR_IMAGE_BLOCK = _env_bool("PADDLE_VL_USE_OCR_FOR_IMAGE_BLOCK")

# Per-task pixel bounds, if any. Empty values mean "no override".
_VLM_EXTRA_ARGS: dict[str, int] = {}
for _key in (
    "ocr_min_pixels",
    "ocr_max_pixels",
    "table_min_pixels",
    "table_max_pixels",
    "chart_min_pixels",
    "chart_max_pixels",
    "formula_min_pixels",
    "formula_max_pixels",
    "seal_min_pixels",
    "seal_max_pixels",
):
    _raw = os.environ.get(f"PADDLE_VL_{_key.upper()}")
    if _raw:
        try:
            _VLM_EXTRA_ARGS[_key] = int(_raw)
        except ValueError:
            logger.warning(
                "Invalid integer for PADDLE_VL_%s=%r; ignoring", _key.upper(), _raw
            )

# Deduplication + cache tunables.
DEDUP_TTL_SECONDS = float(os.environ.get("PADDLE_VL_DEDUP_TTL_SECONDS", "120"))
RESULT_CACHE_MAX = int(os.environ.get("PADDLE_VL_RESULT_CACHE_SIZE", "256"))
RESULT_CACHE_TTL = float(os.environ.get("PADDLE_VL_RESULT_CACHE_TTL", "300"))
MAX_CONCURRENT_INFERENCES = int(os.environ.get("PADDLE_VL_MAX_CONCURRENT", "1"))
# Per-request timeout. Default 3600s matches the backend's
# ``PADDLE_VL_API_TIMEOUT`` in config.py — CPU PaddleOCR-VL can take
# 30–60+ min on dense legal pages, so we don't want to cut inference
# short below the client's patience budget.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("PADDLE_VL_REQUEST_TIMEOUT", "3600"))
SHUTDOWN_GRACE_SECONDS = float(os.environ.get("PADDLE_VL_SHUTDOWN_GRACE", "30"))

security = HTTPBearer(auto_error=False)


# ── PaddleOCR-VL block-type mapping ────────────────────────────────────────
# PaddleOCR-VL-1.5 emits these ``block_label`` values per the official
# docs (https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html):
#
#   text             – normal paragraph text
#   doc_title        – document-level title
#   paragraph_title  – section heading
#   table            – table block (content is Markdown or HTML)
#   formula          – mathematical formula
#   image            – raster figure (content optional)
#   seal             – stamp / seal (new in v1.5)
#   chart            – chart / graph
#   vision_footnote  – footnote text rendered near figure
#   number           – page number (default-ignored in markdown)
#   footnote         – footnote text (default-ignored in markdown)
#   header / footer  – running header / footer text
#   aside_text       – marginalia / sidebar text (default-ignored)
#   header_image / footer_image – running image strips (default-ignored)
#
# Map them to the :class:`ocr.layout_schema.BlockType` vocabulary.
_OFFICIAL_TYPE_TO_BLOCK_TYPE: dict[str, str] = {
    # Headings
    "title": "title",
    "doc_title": "title",
    "paragraph_title": "heading",
    "heading": "heading",
    # Body text
    "text": "paragraph",
    "paragraph": "paragraph",
    "content": "paragraph",
    "formula": "paragraph",
    # Structure
    "table": "table",
    "list": "list",
    "list_item": "list_item",
    # Visual
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "seal": "figure",
    # Repetitive / marginalia
    "header": "header",
    "footer": "footer",
    "vision_footnote": "footer",
}

# Labels that should NOT become LayoutBlocks. These are usually small,
# repetitive, or noise elements that hurt downstream chunking/embedding.
# Mirrors the default ``markdown_ignore_labels`` in PaddleOCR-VL.
_IGNORED_LABELS: frozenset[str] = frozenset(
    {
        "number",
        "footnote",
        "header_image",
        "footer_image",
        "aside_text",
    }
)


# ── Result-shape helpers ───────────────────────────────────────────────────
def _result_json(result: Any) -> dict[str, Any]:
    """Read PaddleOCR's result JSON across supported release shapes.

    ``PaddleOCRVL.predict`` has shipped at least three shapes across
    release branches (``json`` callable returning a dict, ``json`` attr
    holding a JSON string, or a plain ``res`` dict on the top-level
    payload). Normalise all of them to ``dict``.
    """
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _block_bbox(block: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return an official bbox as ``(x, y, width, height)`` in image pixels."""
    raw = next(
        (
            block.get(key)
            for key in ("block_bbox", "bbox", "box", "coordinate")
            if block.get(key) is not None
        ),
        None,
    )
    if isinstance(raw, dict):
        x0 = raw.get("x0", raw.get("left", raw.get("xmin", raw.get("x", 0))))
        y0 = raw.get("y0", raw.get("top", raw.get("ymin", raw.get("y", 0))))
        x1 = raw.get("x1", raw.get("right", raw.get("xmax")))
        y1 = raw.get("y1", raw.get("bottom", raw.get("ymax")))
        if x1 is None or y1 is None:
            x1 = float(x0) + float(raw.get("width", 0))
            y1 = float(y0) + float(raw.get("height", 0))
        raw = [x0, y0, x1, y1]
    # PaddleOCR-VL stores ``block_bbox`` as a numpy array of ``np.float32``
    # in the live ``.json`` attribute (the JSON file form converts these
    # to plain lists). Accept any indexable sequence with length ≥ 4.
    if raw is None or not hasattr(raw, "__getitem__") or len(raw) < 4:
        return 0.0, 0.0, 0.0, 0.0
    try:
        # ``float()`` coerces both Python floats and numpy scalars.
        x0, y0, x1, y1 = (float(v) for v in raw[:4])
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0
    return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)


def _layout_from_official_result(
    output: list[Any],
    page_number: int,
    image_path: Path,
) -> tuple[list[LayoutBlock], list[Any]]:
    """Convert a ``PaddleOCRVL.predict`` output list into LayoutBlocks.

    Keeps PaddleOCR-VL regions rather than flattening them into plain
    paragraphs. Falls back to the Markdown pass if the official
    structured output is empty.
    """
    blocks: list[LayoutBlock] = []
    tables: list[Any] = []
    for result in output:
        data = _result_json(result)
        regions = data.get("parsing_res_list") or data.get("layout_parsing_res") or []
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            text = str(
                region.get("block_content")
                or region.get("text")
                or region.get("content")
                or ""
            ).strip()
            raw_label = str(
                region.get("block_label")
                or region.get("label")
                or region.get("type")
                or "text"
            ).lower()

            # Skip labels that don't belong in the LayoutBlock stream
            # (page numbers, running images, etc.). These mirror
            # PaddleOCR-VL's own ``markdown_ignore_labels`` default.
            if raw_label in _IGNORED_LABELS:
                continue

            block_type = _OFFICIAL_TYPE_TO_BLOCK_TYPE.get(raw_label, "paragraph")
            if not text and block_type != "figure":
                continue
            x, y, width, height = _block_bbox(region)
            html: Optional[str] = None
            if block_type == "table":
                if text.lstrip().lower().startswith("<table"):
                    html = text
                elif _is_markdown_table(text):
                    html = _markdown_table_to_html(text)
                    tables.append(
                        _markdown_to_layout_table(
                            f"p{page_number}_t{len(tables):02d}", text
                        )
                    )
            confidence = region.get("confidence", region.get("score", 0.95))
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.95
            blocks.append(
                LayoutBlock(
                    block_id=f"p{page_number}_b{len(blocks):03d}",
                    type=block_type,
                    text=text,
                    html=html,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    language=_detect_language(text),
                    reading_order=len(blocks) + 1,
                    source_label=raw_label,
                )
            )
    if blocks:
        return blocks, tables
    return _layout_from_markdown(
        _markdown_from_official_result(output, image_path), page_number
    )


def _markdown_from_official_result(output: list[Any], image_path: Path) -> str:
    """Extract the consolidated Markdown string from PaddleOCR-VL output.

    Per the official docs the ``markdown`` attribute is a dict with
    keys ``markdown_texts`` (list[str], one per page), ``markdown_images``
    (list[Image.Image]) and ``page_continuation_flags`` (tuple[bool, ...]).
    We join the text entries for our single-page input.
    """
    markdown_pages: list[str] = []
    for res in output:
        markdown = getattr(res, "markdown", None)
        if isinstance(markdown, dict):
            texts = markdown.get("markdown_texts")
            if isinstance(texts, list):
                for entry in texts:
                    if entry:
                        markdown_pages.append(str(entry))
                if markdown_pages:
                    continue
            text = markdown.get("text") or ""
            if text:
                markdown_pages.append(str(text))
                continue

        data = _result_json(res)
        if data:
            blocks = data.get("parsing_res_list", [])
            text = "\n\n".join(
                str(block.get("block_content", "")).strip()
                for block in blocks
                if str(block.get("block_content", "")).strip()
            )
            if text:
                markdown_pages.append(text)

    markdown_text = "\n\n".join(page.strip() for page in markdown_pages if page.strip())
    if not markdown_text:
        raise RuntimeError(f"PaddleOCR-VL returned no Markdown/text for {image_path}")
    return markdown_text.strip()


def _layout_from_markdown(
    markdown_text: str, page_number: int
) -> tuple[list[LayoutBlock], list[Any]]:
    """Convert plain Markdown text into LayoutBlocks (paragraph/table)."""
    blocks: list[LayoutBlock] = []
    tables: list[Any] = []
    reading_order = 0
    block_idx = 0

    for paragraph in _split_paragraphs(markdown_text):
        text = paragraph.strip()
        if not text:
            continue

        block_id = f"p{page_number}_b{block_idx:03d}"
        block_idx += 1
        reading_order += 1

        if _is_markdown_table(text):
            table_id = f"p{page_number}_t{len(tables):02d}"
            tables.append(_markdown_to_layout_table(table_id, text))
            blocks.append(
                LayoutBlock(
                    block_id=block_id,
                    type="table",
                    text=text,
                    html=_markdown_table_to_html(text),
                    reading_order=reading_order,
                    language=_detect_language(text),
                    confidence=0.95,
                )
            )
        else:
            blocks.append(
                LayoutBlock(
                    block_id=block_id,
                    type="paragraph",
                    text=text,
                    reading_order=reading_order,
                    language=_detect_language(text),
                    confidence=0.95,
                )
            )

    return blocks, tables


# ── Engine wrapper ─────────────────────────────────────────────────────────
class OfficialPaddleVLEngine:
    """Small API adapter around PaddleOCR's official ``PaddleOCRVL`` pipeline.

    Initialised once at startup, then called per request. The model
    itself is thread-safe at the Python level (PaddlePaddle releases the
    GIL around heavy ops); we still serialise calls with
    ``engine_lock`` so two requests can't both contend for VRAM/CPU.
    """

    def __init__(
        self,
        *,
        use_gpu: bool,
        max_new_tokens: int,
        model_cache_dir: Optional[str] = None,
    ) -> None:
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.model_cache_dir = model_cache_dir
        self._pipeline: Any = None
        self._initialized: bool = False
        self._init_lock = asyncio.Lock()

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Lazy / idempotent load of the PaddleOCRVL pipeline.

        Concurrent callers (rare during startup, common during k8s
        readiness probes) serialise on ``_init_lock`` so the model is
        loaded exactly once.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            from paddleocr import PaddleOCRVL

            init_kwargs: dict[str, Any] = {
                "pipeline_version": "v1.5",
                "vl_rec_model_name": "PaddleOCR-VL-1.5-0.9B",
                "device": PADDLE_OCR_DEVICE,
            }

            # Inference engine. The transformers backend is required
            # for 8-bit / 4-bit quantisation (bitsandbytes). When neither
            # 8-bit nor 4-bit is requested we stay on the default
            # PaddlePaddle backend — it's faster on CPU.
            if LOAD_IN_8BIT or LOAD_IN_4BIT:
                init_kwargs["engine"] = "transformers"
                if LOAD_IN_4BIT:
                    init_kwargs["load_in_4bit"] = True
                else:
                    init_kwargs["load_in_8bit"] = True
            elif PIPELINE_ENGINE:
                init_kwargs["engine"] = PIPELINE_ENGINE

            # VLM inference service separation. When set, PaddleOCR-VL
            # runs the layout analysis locally and delegates only the
            # VLM stage to the configured server. This is the
            # recommended production deployment topology on GPU
            # hardware.
            if VL_REC_BACKEND:
                init_kwargs["vl_rec_backend"] = VL_REC_BACKEND
            if VL_REC_SERVER_URL:
                init_kwargs["vl_rec_server_url"] = VL_REC_SERVER_URL
            if VL_REC_MAX_CONCURRENCY is not None:
                init_kwargs["vl_rec_max_concurrency"] = VL_REC_MAX_CONCURRENCY
            if VL_REC_API_MODEL_NAME:
                init_kwargs["vl_rec_api_model_name"] = VL_REC_API_MODEL_NAME
            if VL_REC_API_KEY:
                init_kwargs["vl_rec_api_key"] = VL_REC_API_KEY

            model_dir = os.environ.get("PADDLE_VL_MODEL_DIR") or self.model_cache_dir
            if model_dir:
                model_path = Path(model_dir).expanduser()
                if model_path.exists():
                    init_kwargs["vl_rec_model_dir"] = str(model_path)
                else:
                    logger.warning(
                        "PaddleOCR-VL model dir does not exist: %s", model_path
                    )

            # Document pre-processing (orientation classification + deskew).
            doc_preprocess = _env_bool("PADDLE_VL_USE_DOC_PREPROCESSOR")
            if doc_preprocess is not None:
                init_kwargs["use_doc_orientation_classify"] = doc_preprocess
                init_kwargs["use_doc_unwarping"] = doc_preprocess

            # Layout analysis. We default to True for the full
            # pipeline; users can opt out for raw VLM-only inference.
            layout_detection = _env_bool("PADDLE_VL_USE_LAYOUT_DETECTION")
            if layout_detection is not None:
                init_kwargs["use_layout_detection"] = layout_detection

            # Default to formatted markdown block_content — the
            # LayoutBlock consumers downstream expect the markdown-form
            # of tables / formulas.
            format_block_content = _env_bool("PADDLE_VL_FORMAT_BLOCK_CONTENT")
            if format_block_content is not None:
                init_kwargs["format_block_content"] = format_block_content
            else:
                init_kwargs["format_block_content"] = True

            merge_layout_blocks = _env_bool("PADDLE_VL_MERGE_LAYOUT_BLOCKS")
            if merge_layout_blocks is not None:
                init_kwargs["merge_layout_blocks"] = merge_layout_blocks

            # v1.5 feature toggles.
            if USE_CHART_RECOGNITION is not None:
                init_kwargs["use_chart_recognition"] = USE_CHART_RECOGNITION
            if USE_SEAL_RECOGNITION is not None:
                init_kwargs["use_seal_recognition"] = USE_SEAL_RECOGNITION
            if USE_OCR_FOR_IMAGE_BLOCK is not None:
                init_kwargs["use_ocr_for_image_block"] = USE_OCR_FOR_IMAGE_BLOCK

            # v1.5 layout shape mode (rect / quad / poly / auto).
            if LAYOUT_SHAPE_MODE:
                init_kwargs["layout_shape_mode"] = LAYOUT_SHAPE_MODE

            # VLM sampling params (deterministic OCR by default).
            if TEMPERATURE is not None:
                init_kwargs["temperature"] = TEMPERATURE
            if TOP_P is not None:
                init_kwargs["top_p"] = TOP_P
            if REPETITION_PENALTY is not None:
                init_kwargs["repetition_penalty"] = REPETITION_PENALTY

            # Image preprocessing bounds.
            if MIN_PIXELS is not None:
                init_kwargs["min_pixels"] = MIN_PIXELS
            if MAX_PIXELS is not None:
                init_kwargs["max_pixels"] = MAX_PIXELS

            # Per-task pixel bounds via ``vlm_extra_args``.
            if _VLM_EXTRA_ARGS:
                init_kwargs["vlm_extra_args"] = dict(_VLM_EXTRA_ARGS)

            # CPU threading. The k8s deployment sets
            # ``OMP_NUM_THREADS=8`` and ``MKL_NUM_THREADS=8`` at the
            # container level. Pass it through so PaddleOCR's internal
            # MKL-DNN cache picks the right thread count.
            cpu_threads = os.environ.get("PADDLE_VL_CPU_THREADS") or os.environ.get(
                "OMP_NUM_THREADS"
            )
            if cpu_threads:
                try:
                    init_kwargs["cpu_threads"] = int(cpu_threads)
                except ValueError:
                    logger.warning(
                        "Invalid PADDLE_VL_CPU_THREADS=%r; using framework default",
                        cpu_threads,
                    )

            logger.info(
                "Loading official PaddleOCR-VL pipeline (device=%s, model_dir=%s, "
                "engine=%s, layout=%s, format_block_content=%s, chart=%s, "
                "seal=%s, shape=%s, cpu_threads=%s)",
                init_kwargs["device"],
                init_kwargs.get("vl_rec_model_dir", "official cache/download"),
                init_kwargs.get("engine", "default"),
                init_kwargs.get("use_layout_detection", "default"),
                init_kwargs.get("format_block_content", "default"),
                init_kwargs.get("use_chart_recognition", "default"),
                init_kwargs.get("use_seal_recognition", "default"),
                init_kwargs.get("layout_shape_mode", "default"),
                init_kwargs.get("cpu_threads", "default"),
            )
            t0 = time.perf_counter()
            # PaddleOCRVL's constructor is blocking. Run it in a worker
            # thread so the event loop stays responsive during k8s
            # readiness probes.
            self._pipeline = await asyncio.to_thread(
                _build_paddleocrvl_pipeline, init_kwargs
            )
            self._initialized = True
            logger.info(
                "PaddleOCR-VL official pipeline ready in %.1f s",
                time.perf_counter() - t0,
            )

    def cleanup(self) -> None:
        """Drop the pipeline reference so Python can GC it on shutdown."""
        self._pipeline = None
        self._initialized = False

    def extract_page(self, image: Image.Image, page_number: int = 1) -> LayoutPage:
        """Run OCR on a single image and return a :class:`LayoutPage`."""
        if not self._initialized:
            raise RuntimeError("PaddleOCR-VL pipeline is not initialized")
        if self._pipeline is None:
            raise RuntimeError("PaddleOCR-VL pipeline is None after init")

        rgb_image = image.convert("RGB")
        page_w, page_h = rgb_image.size

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            rgb_image.save(tmp, format="PNG")

        try:
            # ``predict`` may return a generator; materialise it once
            # because structured regions are inspected before any
            # Markdown fallback.
            #
            # NOTE: ``prompt_label`` is only honoured when
            # ``use_layout_detection=False``. We forward it
            # unconditionally so users who disable layout detection can
            # steer the VLM via ``PADDLE_VL_PROMPT_LABEL``. With the
            # default full-pipeline mode (``use_layout_detection=True``)
            # the parameter is ignored by PaddleOCR-VL.
            predict_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
            }
            if PROMPT_LABEL:
                predict_kwargs["prompt_label"] = PROMPT_LABEL
            output = list(self._pipeline.predict(str(tmp_path), **predict_kwargs))
            blocks, tables = _layout_from_official_result(
                output, page_number, tmp_path
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                logger.debug(
                    "Failed to remove temporary OCR image: %s",
                    tmp_path,
                    exc_info=True,
                )
            # Release the PIL Image handle explicitly. PIL keeps a
            # buffer cache; not freeing it here can balloon RSS under
            # sustained load.
            try:
                rgb_image.close()
            except Exception:  # pragma: no cover
                pass

        return LayoutPage(
            page_number=page_number,
            width_px=page_w,
            height_px=page_h,
            dpi=0,
            blocks=blocks,
            tables=tables,
        )


def _build_paddleocrvl_pipeline(init_kwargs: dict[str, Any]) -> Any:
    """Module-level helper so ``asyncio.to_thread`` can target a real callable."""
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(**init_kwargs)


# ── Cache + dedup primitives ───────────────────────────────────────────────
@dataclass
class _CacheEntry:
    """One cached response payload, bounded by TTL and size.

    The cache key lives on the :class:`collections.OrderedDict` that
    stores the entry, so we deliberately don't duplicate it here.
    """

    payload: dict[str, Any]
    expires_at: float


class _ResponseCache:
    """Bounded LRU+TTL cache mapping request_key → response payload.

    We use ``collections.OrderedDict`` so the eviction policy is
    deterministic (oldest by insertion, oldest-by-touch is implemented
    by ``move_to_end``). The default policy is FIFO eviction at capacity
    which matches "oldest cached response wins" — good enough for short
    retry windows.
    """

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max(1, int(max_size))
        self._ttl = max(0.0, float(ttl_seconds))
        self._entries: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        if self._ttl <= 0:
            return None
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                self._entries.pop(key, None)
                return None
            # LRU touch
            self._entries.move_to_end(key)
            return entry.payload

    async def put(self, key: str, payload: dict[str, Any]) -> None:
        if self._ttl <= 0 or self._max_size <= 0:
            return
        async with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = _CacheEntry(
                payload=payload,
                expires_at=time.monotonic() + self._ttl,
            )
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()


class _InflightTracker:
    """Track currently-running VLM tasks so retries can join them.

    Each entry holds an :class:`asyncio.Future` that resolves to the
    response payload (or raises). The tracker enforces a TTL on
    *parked* futures so a wedged worker can't pin an entry forever; if
    the worker process dies, the TTL ensures the dict self-heals.

    The actual VLM inference is *not* awaited under this lock — only
    dict mutation is. That way one slow inference can't block another
    request's lookup.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = max(0.0, float(ttl_seconds))
        self._futures: "dict[str, tuple[asyncio.Future[dict[str, Any]], float]]" = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, key: str
    ) -> tuple[asyncio.Future[dict[str, Any]], bool]:
        """Return ``(future, is_new)``.

        If an unexpired future exists for ``key``, return it with
        ``is_new=False``. Otherwise create a new pending future and
        return it with ``is_new=True``.
        """
        now = time.monotonic()
        async with self._lock:
            existing = self._futures.get(key)
            if existing is not None:
                fut, expires_at = existing
                if expires_at > now and not fut.done():
                    return fut, False
                # Stale or done — drop it; the caller will create a fresh one.
                self._futures.pop(key, None)
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._futures[key] = (fut, now + self._ttl)
            return fut, True

    async def resolve(self, key: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            entry = self._futures.pop(key, None)
        if entry is None:
            return
        fut, _expires = entry
        if not fut.done():
            fut.set_result(payload)

    async def fail(self, key: str, exc: BaseException) -> None:
        async with self._lock:
            entry = self._futures.pop(key, None)
        if entry is None:
            return
        fut, _expires = entry
        if not fut.done():
            fut.set_exception(exc)

    async def sweep_expired(self) -> int:
        """Remove expired entries and fail their futures.

        Returns the number evicted.  Any still-pending future whose TTL
        has passed receives a ``RuntimeError`` so joined waiters don't
        hang forever.
        """
        if self._ttl <= 0:
            return 0
        now = time.monotonic()
        expired_futures: list[asyncio.Future[dict[str, Any]]] = []
        async with self._lock:
            expired = [k for k, (_, exp) in self._futures.items() if exp <= now]
            for k in expired:
                entry = self._futures.pop(k, None)
                if entry is not None:
                    expired_futures.append(entry[0])
        # Fail any still-pending futures *outside* the lock so joined
        # waiters wake up immediately with a clear error.
        for fut in expired_futures:
            if not fut.done():
                fut.set_exception(
                    RuntimeError(
                        f"Inflight OCR entry expired after {self._ttl:.0f}s"
                    )
                )
        return len(expired)

    async def active_count(self) -> int:
        async with self._lock:
            return sum(1 for fut, _ in self._futures.values() if not fut.done())

    async def all_futures(self) -> list[asyncio.Future[dict[str, Any]]]:
        async with self._lock:
            return [fut for fut, _ in self._futures.values()]

    async def clear(self) -> None:
        async with self._lock:
            self._futures.clear()


# ── Auth ───────────────────────────────────────────────────────────────────
def verify_api_key(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """Verify ``OCR_API_KEY`` via Bearer or Basic auth.

    If ``OCR_API_KEY`` is unset, the service is open (useful inside
    a private k8s network). When set, only requests with a matching
    credential get through.
    """
    if not API_KEY:
        return None

    header = request.headers.get("Authorization")
    if not header:
        raise HTTPException(
            status_code=403, detail="Invalid authentication credentials"
        )

    if header.startswith("Bearer "):
        token = header.split(" ", 1)[1].strip()
        if token == API_KEY:
            return token

    if header.startswith("Basic "):
        try:
            import base64

            decoded = base64.b64decode(header.split(" ", 1)[1]).decode()
            if API_KEY in decoded.split(":"):
                return API_KEY
        except Exception:  # pragma: no cover
            pass

    logger.warning("Unauthorized access attempt from %s", request.client)
    raise HTTPException(
        status_code=403, detail="Invalid authentication credentials"
    )


# ── Global state (initialised in lifespan) ────────────────────────────────
engine: Optional[OfficialPaddleVLEngine] = None
engine_lock: Optional[asyncio.Semaphore] = None  # limits concurrent VLM calls
inflight: Optional[_InflightTracker] = None
response_cache: Optional[_ResponseCache] = None
_sweep_task: Optional[asyncio.Task[None]] = None


def _build_request_key(
    image_bytes: bytes, language: str, layout: str, model: str
) -> str:
    """Derive a short, stable cache key for a request.

    We hash the image content + form fields so identical retries
    collapse to a single VLM call. The hash is hex (32-128 chars
    depending on backend) — short enough for logs, collision-safe in
    practice.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(image_bytes)
    h.update(b"\x00")
    h.update(language.encode("utf-8"))
    h.update(b"\x00")
    h.update(layout.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _build_elements(layout_page: LayoutPage) -> list[dict[str, Any]]:
    """Translate a :class:`LayoutPage` into the wire-format ``elements`` list.

    Matches the schema documented in
    :mod:`ocr.paddle_vl_remote_engine`. Each element is one of:

    * ``type``     — ``paragraph`` / ``title`` / ``heading`` / ``table`` /
                     ``list`` / ``list_item`` / ``figure`` / ``header`` /
                     ``footer``.
    * ``text``     — extracted text (empty for pure figures).
    * ``html``     — HTML representation for tables (``None`` otherwise).
    * ``bbox``     — ``[x0, y0, x1, y1]`` in pixels, or ``None`` if
                     geometry is unknown.
    * ``confidence`` — engine confidence (0..1).
    * ``language`` — ``ta`` / ``en`` / ``mixed`` / ``unknown``.
    """
    elements: list[dict[str, Any]] = []
    for block in layout_page.blocks:
        element_type = block.type
        # Map "heading" Visual Block to "title" PageElement (matches the
        # convention the remote-engine client expects).
        if element_type == "heading":
            element_type = "title"

        bbox: Optional[list[float]] = None
        if block.width > 0 or block.height > 0:
            bbox = [
                block.x,
                block.y,
                block.x + block.width,
                block.y + block.height,
            ]

        elements.append(
            {
                "type": element_type,
                "text": block.text,
                "html": block.html,
                "bbox": bbox,
                "confidence": block.confidence,
                "language": block.language,
            }
        )
    return elements


async def _sweep_loop(interval: float) -> None:
    """Background task that prunes expired inflight entries.

    Runs every ``interval`` seconds. If the event loop is shutting
    down, the task is cancelled by uvicorn; otherwise it lives for
    the lifetime of the process.
    """
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        if inflight is None:
            return
        try:
            evicted = await inflight.sweep_expired()
            if evicted:
                logger.info("Pruned %d expired inflight entries", evicted)
        except Exception:  # pragma: no cover
            logger.exception("Inflight sweep failed")


# ── FastAPI app + lifespan ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise engine + caches on startup, drain on shutdown."""
    global engine, engine_lock, inflight, response_cache, _sweep_task

    # ``Semaphore`` (not ``Lock``) lets us cap *parallel* VLM calls.
    # The default of 1 means "one inference at a time" which is the
    # safest choice for CPU-bound 7B-class models on a single pod.
    engine_lock = asyncio.Semaphore(max(1, MAX_CONCURRENT_INFERENCES))
    inflight = _InflightTracker(REQUEST_TIMEOUT_SECONDS)
    response_cache = _ResponseCache(RESULT_CACHE_MAX, RESULT_CACHE_TTL)

    if engine is None:
        logger.info(
            "Initializing PaddleVLEngine (GPU=%s, 8bit=%s, 4bit=%s, "
            "max_concurrent=%d, request_timeout=%.0fs) ...",
            USE_GPU,
            LOAD_IN_8BIT,
            LOAD_IN_4BIT,
            MAX_CONCURRENT_INFERENCES,
            REQUEST_TIMEOUT_SECONDS,
        )
        try:
            engine = OfficialPaddleVLEngine(
                use_gpu=USE_GPU,
                max_new_tokens=MAX_NEW_TOKENS,
                model_cache_dir=CACHE_DIR,
            )
            await engine.initialize()
            logger.info("PaddleVLEngine loaded and initialized successfully!")
        except Exception as exc:
            logger.exception("Failed to load model during startup")
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    # Sweep at DEDUP_TTL / 4 but cap at 60 s so the loop stays
    # responsive even when REQUEST_TIMEOUT is very large.
    sweep_interval = max(5.0, min(DEDUP_TTL_SECONDS / 4.0, 60.0))
    _sweep_task = asyncio.create_task(_sweep_loop(sweep_interval))

    try:
        yield
    finally:
        # Drain inflight requests up to SHUTDOWN_GRACE so we don't kill
        # mid-inference workers on rolling deploys.
        if _sweep_task is not None:
            _sweep_task.cancel()
            try:
                await _sweep_task
            except (asyncio.CancelledError, Exception):
                pass
            _sweep_task = None

        if inflight is not None:
            futures = await inflight.all_futures()
            if futures:
                logger.info(
                    "Draining %d inflight OCR requests (grace=%.1fs) ...",
                    len(futures),
                    SHUTDOWN_GRACE_SECONDS,
                )
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*futures, return_exceptions=True),
                        timeout=SHUTDOWN_GRACE_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Shutdown grace exceeded; cancelling remaining requests"
                    )
                    for fut in futures:
                        if not fut.done():
                            fut.cancel()

        if response_cache is not None:
            await response_cache.clear()
        if inflight is not None:
            await inflight.clear()

        if engine is not None:
            logger.info("Cleaning up engine...")
            engine.cleanup()
            engine = None


app = FastAPI(
    title="Local PaddleOCR-VL-1.5 API",
    version="1.5.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so unhandled errors return JSON, never an HTML stack page."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc) or exc.__class__.__name__,
        },
    )


# ── Health & readiness ────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Liveness probe — process is up.

    Returns ``status="healthy"`` only when the model is loaded. While
    the model is loading we return ``"loading"`` so k8s readiness
    probes don't route traffic prematurely.
    """
    is_ready = engine is not None and engine.initialized
    inflight_count = await inflight.active_count() if inflight is not None else 0
    return {
        "status": "healthy" if is_ready else "loading",
        "model": "PaddleOCR-VL-1.5",
        "engine": "paddle_vl",
        "gpu_enabled": USE_GPU,
        "device": PADDLE_OCR_DEVICE,
        "load_in_8bit": LOAD_IN_8BIT,
        "load_in_4bit": LOAD_IN_4BIT,
        "inflight_requests": inflight_count,
        "max_concurrent_inferences": MAX_CONCURRENT_INFERENCES,
    }


@app.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness probe — model is warm and ready to serve traffic."""
    if engine is None or not engine.initialized:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "model not initialised"},
        )
    return {"status": "ready"}


# ── OCR endpoint ──────────────────────────────────────────────────────────
@app.post("/ocr")
@app.post("/v1/ocr")
async def process_image_ocr(
    image: UploadFile = File(...),
    language: str = Form("auto"),
    layout: str = Form("true"),
    model: str = Form("paddle_vl"),
    _api_key: Optional[str] = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run PaddleOCR-VL on a single uploaded image.

    Request (multipart/form-data)::

        image    : <PNG/JPEG bytes>
        language : "auto" | "en" | "ta" ...
        layout   : "true" | "false"
        model    : "paddle_vl"

    Response::

        {
          "engine":   "paddle_vl",
          "language": "auto",
          "elements": [
            {"type": "...", "text": "...", "html": null|str,
             "bbox": [x0,y0,x1,y1] | null,
             "confidence": 0.95, "language": "ta"|"en"|"mixed"|"unknown"}
          ],
          "request_key": "...",
          "cached":     true|false,
          "elapsed_ms": 1234
        }

    Concurrency contract
    --------------------
    * Identical requests within ``PADDLE_VL_DEDUP_TTL_SECONDS`` (default
      120 s) share a single VLM inference — the second caller joins the
      first caller's task. Identical responses are cached for
      ``PADDLE_VL_RESULT_CACHE_TTL`` (default 300 s) so quick retries
      skip the VLM entirely.
    * At most ``PADDLE_VL_MAX_CONCURRENT`` (default 1) inferences run
      in parallel; additional callers queue.
    * A single inference cannot run longer than
      ``PADDLE_VL_REQUEST_TIMEOUT`` (default 1800 s).
    """
    if engine is None or not engine.initialized:
        raise HTTPException(
            status_code=503, detail="Model engine not ready or failed to load"
        )
    assert engine_lock is not None
    assert inflight is not None and response_cache is not None

    start_time = time.perf_counter()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    request_key = _build_request_key(image_bytes, language, layout, model)

    # ── 1. Result cache hit — return without re-running VLM ─────────
    cached = await response_cache.get(request_key)
    if cached is not None:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "OCR cache hit: key=%s elapsed_ms=%d",
            request_key[:12],
            elapsed_ms,
        )
        return {**cached, "cached": True, "elapsed_ms": elapsed_ms}

    # ── 2. Join an inflight task for this exact key, or create one ──
    fut, is_new = await inflight.get_or_create(request_key)
    if not is_new:
        logger.info(
            "Joining inflight OCR task: key=%s file=%s size=%d",
            request_key[:12],
            image.filename,
            len(image_bytes),
        )
        try:
            payload = await asyncio.wait_for(
                asyncio.shield(fut), timeout=REQUEST_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"PaddleOCR-VL inference exceeded "
                    f"{REQUEST_TIMEOUT_SECONDS:.0f}s timeout"
                ),
            )
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {**payload, "cached": False, "elapsed_ms": elapsed_ms}

    # ── 3. New request — drive the VLM under the engine semaphore ──
    logger.info(
        "Received OCR request: key=%s file=%s size=%d bytes language=%s layout=%s",
        request_key[:12],
        image.filename,
        len(image_bytes),
        language,
        layout,
    )

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        payload = await asyncio.wait_for(
            _run_inference(pil_image, language),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "OCR request timed out: key=%s timeout=%.0fs",
            request_key[:12],
            REQUEST_TIMEOUT_SECONDS,
        )
        await inflight.fail(
            request_key,
            RuntimeError(
                f"PaddleOCR-VL inference exceeded "
                f"{REQUEST_TIMEOUT_SECONDS:.0f}s timeout"
            ),
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"PaddleOCR-VL inference exceeded "
                f"{REQUEST_TIMEOUT_SECONDS:.0f}s timeout"
            ),
        )
    except HTTPException:
        await inflight.fail(request_key, RuntimeError("client error"))
        raise
    except Exception as exc:
        logger.exception("OCR execution failed: key=%s", request_key[:12])
        await inflight.fail(request_key, exc)
        raise HTTPException(
            status_code=500, detail=f"OCR execution failed: {exc}"
        )
    else:
        await inflight.resolve(request_key, payload)
        await response_cache.put(request_key, payload)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "Successfully processed OCR request: key=%s elements=%d elapsed_ms=%d",
            request_key[:12],
            len(payload.get("elements", [])),
            elapsed_ms,
        )
        return {**payload, "cached": False, "elapsed_ms": elapsed_ms}


async def _run_inference(pil_image: Image.Image, language: str) -> dict[str, Any]:
    """Acquire the engine semaphore and run a single OCR pass in a worker thread."""
    assert engine_lock is not None and engine is not None
    async with engine_lock:
        # ``asyncio.to_thread`` releases the event loop while the
        # CPU/GPU-bound PaddleOCRVL inference runs.
        layout_page = await asyncio.to_thread(engine.extract_page, pil_image)

    elements = _build_elements(layout_page)
    return {
        "engine": "paddle_vl",
        "language": language,
        "elements": elements,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Start PaddleOCR-VL-1.5 API service")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8001, help="Listen port")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HuggingFace model cache dir",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["PADDLE_VL_CACHE_DIR"] = args.cache_dir

    uvicorn.run(
        "paddle_vl_api:app",
        host=args.host,
        port=args.port,
        # Conservative defaults for VLM workloads — long-lived
        # connections, no per-request body size surprise.
        timeout_keep_alive=300,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
