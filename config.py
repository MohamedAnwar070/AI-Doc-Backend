"""Application configuration.

Centralised, pydantic-settings driven configuration. Values are resolved from
(in order): environment variables → a local ``.env`` file → defaults below.

Two principles are enforced for production / SonarQube readiness:

1.  **No hidden magic numbers** — every tunable knob is declared here with a
    type hint and a sensible default.
2.  **Models load from mounted paths or remote endpoints.** OCR / LLM may stay
    remote; **query embeddings run in-process by default** (see
    ``EMBEDDING_RUNTIME`` / ``EMBEDDING_LOCAL_MODEL_PATH``) because remote
    embedding is the dominant voice-turn cost. Remote embedding remains a
    fallback when the local model is unavailable.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    All fields support the corresponding upper-case environment variable
    (e.g. ``OCR_ENGINE`` for :pyattr:`ocr_engine`). The ``GPU_ENABLED`` flag
    is honoured by every engine that supports CUDA.
    """

    # ── OCR engine selection ──────────────────────────────────────────────
    # paddle      → local PaddleOCR (Tamil + English).
    # paddle_vl   → PaddleOCR-VL-1.5 via HuggingFace transformers (VLM, GPU required).
    # remote      → HTTP client to an externally-hosted OCR microservice
    #               (see ``OCR_REMOTE_URL``). Recommended for production.
    # google_docai → Google Cloud Document AI. Best quality for bilingual
    #               scanned PDFs. Requires ``GOOGLE_DOCAI_PROCESSOR``.
    ocr_engine: Literal["paddle", "paddle_vl", "paddle_vl_ft", "olm_ocr", "remote", "google_docai", "google_visionai", "sur_ocr", "ocr_open_ai", "PaddleOCR1.5"] = Field(
        default="paddle",
        description="OCR backend implementation to use.",
    )
    ocr_lang: str = Field(
        default="auto",
        description="Default OCR language. 'auto' detects per-document.",
    )

    # ── Hardware ──────────────────────────────────────────────────────────
    # GPU_ENABLED is the canonical switch. ``use_gpu`` is kept as an alias
    # so older code paths and Helm charts keep working.
    gpu_enabled: bool = Field(default=False, description="Enable CUDA GPU.")
    use_gpu: bool = Field(default=False, description="Deprecated: alias of GPU_ENABLED.")

    # ── Processing ────────────────────────────────────────────────────────
    max_workers: int = Field(default=4, ge=1, le=64)
    batch_size: int = Field(default=10, ge=1)
    ocr_dpi: int = Field(
        default=350,
        ge=200,
        le=600,
        description="PDF render DPI. 350 is the sweet spot for Tamil glyphs.",
    )
    ocr_dpi_tamil: int = Field(
        default=400,
        ge=200,
        le=600,
        description="Higher DPI used when a page is detected as Tamil.",
    )

    # ── PP-Structure layout (Paddle classic only) ─────────────────────────
    use_layout_detection: bool = True
    layout_recovery: bool = True
    layout_table_as_html: bool = True

    # ── PaddleOCR-VL-1.5 (transformers VLM) ──────────────────────────────
    # Used when OCR_ENGINE=paddle_vl.
    paddle_vl_load_in_8bit: bool = Field(
        default=False,
        description=(
            "Use bitsandbytes 8-bit quantisation for PaddleOCR-VL-1.5. "
            "Saves ~6 GB VRAM at ~5% quality cost."
        ),
    )
    paddle_vl_load_in_4bit: bool = Field(
        default=False,
        description=(
            "Use bitsandbytes 4-bit quantisation for PaddleOCR-VL-1.5. "
            "Saves ~10 GB VRAM at ~10% quality cost."
        ),
    )
    paddle_vl_max_new_tokens: int = Field(
        default=4096,
        ge=512,
        le=8192,
        description="Max generation tokens per VLM inference call.",
    )
    paddle_vl_cache_dir: str = Field(
        default="",
        description=(
            "Local directory for HuggingFace model cache. "
            "Leave empty to use the default HF_HOME path."
        ),
    )

    # ── PaddleOCR-VL-1.5 Remote API (deployed microservice) ──────────────
    # When PADDLE_VL_API_URL is set, OCR_ENGINE=PaddleOCR1.5 delegates VLM
    # inference to the deployed paddle_vl_api.py service instead of loading
    # the model locally.  This keeps the backend container CPU-only.
    paddle_vl_api_url: str = Field(
        default="",
        description=(
            "Base URL of the deployed PaddleOCR-VL-1.5 API microservice "
            "(e.g. https://paddle-vl.example.com).  When set, OCR_ENGINE=PaddleOCR1.5 "
            "uses this remote API instead of loading the model locally."
        ),
    )
    paddle_vl_api_key: str = Field(
        default="",
        description=(
            "Bearer token / API key for the remote PaddleOCR-VL API. "
            "Must match the OCR_API_KEY env var on the deployed service."
        ),
    )
    paddle_vl_api_timeout: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description=(
            "HTTP timeout in seconds for remote PaddleOCR-VL API calls. "
            "VLM inference on CPU can take 30–60 s per page; set high enough."
        ),
    )

    # ── olmOCR VLM (transformers & API backends) ──────────────────────────
    # Used when OCR_ENGINE=olm_ocr.
    olm_ocr_provider: Literal["local", "openai"] = Field(
        default="local",
        description=(
            "VLM backend provider. 'local' runs the actual olmOCR model (allenai/olmOCR-2-7B) locally via transformers. "
            "'openai' is a fallback that runs vision-based OCR via OpenAI API."
        ),
    )
    olm_ocr_model_id: str = Field(
        default="allenai/olmOCR-2-7B-1025-FP8",
        description="HuggingFace model ID for olmOCR (used in local mode).",
    )
    olm_ocr_load_in_8bit: bool = Field(
        default=False,
        description=(
            "Use bitsandbytes 8-bit quantisation for local olmOCR. "
            "Saves ~6 GB VRAM at ~5% quality cost."
        ),
    )
    olm_ocr_load_in_4bit: bool = Field(
        default=False,
        description=(
            "Use bitsandbytes 4-bit quantisation for local olmOCR. "
            "Saves ~10 GB VRAM at ~10% quality cost."
        ),
    )
    olm_ocr_max_new_tokens: int = Field(
        default=4096,
        ge=512,
        le=8192,
        description="Max generation tokens per VLM inference call.",
    )
    olm_ocr_path: str = Field(
        default="",
        description=(
            "Local directory where the olmOCR model is downloaded. "
            "Set via OLM_OCR_PATH in .env after running download_olmocr_model.py."
        ),
    )
    olm_ocr_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature for VLM inference.",
    )
    olm_ocr_api_model: str = Field(
        default="",
        description="Model override for OpenAI API provider. Defaults to 'gpt-4o'.",
    )

    # ── Image preprocessing pipeline (engine-agnostic) ────────────────────
    enable_preprocessing: bool = Field(
        default=True,
        description="Master switch for the deskew / denoise / binarise pipeline.",
    )
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_binarize: bool = Field(
        default=False,
        description=(
            "Adaptive Sauvola binarisation. Disabled by default because it "
            "can erode delicate Tamil glyph strokes; enable for clean scans."
        ),
    )
    enable_border_crop: bool = True
    enable_clahe: bool = Field(
        default=True,
        description=(
            "Contrast Limited Adaptive Histogram Equalisation. Helps Tamil "
            "scans where ink density varies across the page."
        ),
    )

    # ── Tamil-aware text post-processing ──────────────────────────────────
    tamil_post_process: bool = Field(
        default=True,
        description="Apply NFC + Tamil confusable cleanup to OCR output.",
    )
    tamil_dictionary_correct: bool = Field(
        default=True,
        description=(
            "Apply a small set of unambiguous Tamil OCR substitutions "
            "(e.g. zero-width fixes, common Paddle-Tamil mis-recognitions). "
            "Always conservative; never rewrites legitimate Tamil words."
        ),
    )

    # ── Per-block translation (optional, requires `transformers`) ─────────
    enable_translation: bool = False
    translation_target_lang: str = "en"

    # ── LLM-based OCR text cleaning ──────────────────────────────────────
    # When enabled, every extracted page is sent to the configured LLM
    # (AI_MODEL_URL) for OCR-noise correction *before* being stored.
    # The model fixes garbled characters / broken words but preserves
    # meaning, structure, and bilingual content.
    llm_clean_enabled: bool = Field(
        default=False,
        description=(
            "Post-process each OCR page through the LLM to fix OCR noise. "
            "Uses AI_MODEL_URL / AI_MODEL_AUTH / AI_MODEL_MODEL."
        ),
    )
    llm_clean_max_chars: int = Field(
        default=4000,
        ge=500,
        le=16000,
        description=(
            "Maximum characters per LLM-clean request. Longer pages are "
            "split into chunks to stay within model context limits."
        ),
    )

    # ── Native PDF text extraction ─────────────────────────────────────────
    native_strip_header_pct: float = Field(
        default=0.00,
        ge=0.0,
        le=0.25,
        description=(
            "For native/text-layer PDFs: strip text blocks whose bottom edge "
            "is within this percentage of the page top (running headers). "
            "Applied from page 2 onward so the first-page title is preserved."
        ),
    )
    native_strip_footer_pct: float = Field(
        default=0.00,
        ge=0.0,
        le=0.25,
        description=(
            "For native/text-layer PDFs: strip text blocks whose top edge "
            "is within this percentage of the page bottom (running footers)."
        ),
    )

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", "8001"))
    reload: bool = False

    # ── Storage ───────────────────────────────────────────────────────────
    upload_dir: str = "uploads"
    output_dir: str = "outputs"
    max_upload_size_mb: int = 100

    # ── Office source preview ─────────────────────────────────────────────
    soffice_path: str = Field(
        default="",
        description=(
            "Optional path to LibreOffice soffice executable. Used to render "
            "DOCX/XLSX source previews as PDF for the right-side viewer."
        ),
    )
    office_preview_timeout_sec: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for headless LibreOffice source-preview conversion.",
    )

    # ── Categories ────────────────────────────────────────────────────────
    category_max_depth: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum allowed nested category depth.",
    )

    # ── Google Cloud Document AI ──────────────────────────────────────────
    # Used when ``OCR_ENGINE=google_docai``. Requires a GCP service account
    # with the Document AI API enabled and a Document OCR processor created
    # in the GCP console → Document AI → Processors → "Document OCR".
    google_docai_credentials: str = Field(
        default="",
        description=(
            "Path to a Google Cloud service-account JSON key file. "
            "If empty, falls back to GOOGLE_APPLICATION_CREDENTIALS env var."
        ),
    )
    google_docai_processor: str = Field(
        default="",
        description=(
            "Full resource name of the Document AI processor, e.g. "
            "'projects/my-project/locations/us/processors/abcdef123'. "
            "Takes priority over the split GOOGLE_DOCAI_PROJECT_ID / "
            "GOOGLE_DOCAI_LOCATION / GOOGLE_DOCAI_PROCESSOR_ID vars."
        ),
    )
    # ── Fallback split vars (used only when google_docai_processor is empty) ──
    google_docai_project_id: str = Field(
        default="",
        description="GCP project ID (fallback when full processor name not set).",
    )
    google_docai_location: str = Field(
        default="us",
        description="DocAI location — 'us' or 'eu' (fallback).",
    )
    google_docai_processor_id: str = Field(
        default="",
        description="Bare processor ID (fallback when full processor name not set).",
    )
    google_docai_processor_version: str = Field(
        default="",
        description="Optional processor version ID to pin for reproducibility.",
    )
    google_docai_language_hints: str = Field(
        default="en,ta",
        description=(
            "Comma-separated BCP-47 language codes for OCR hints, "
            "e.g. 'en,ta' for bilingual English + Tamil."
        ),
    )
    google_docai_timeout: float = Field(
        default=300.0,
        ge=5.0,
        le=3600.0,
        description="Timeout in seconds for each Document AI request.",
    )

    # ── DocAI quality / production knobs ──────────────────────────────────
    google_docai_native_pdf: bool = Field(
        default=True,
        description=(
            "Send raw PDF bytes to DocAI (mime_type=application/pdf) instead "
            "of a pre-rendered PNG. DocAI renders internally at optimal DPI — "
            "always better quality and lower latency than our own render."
        ),
    )
    google_docai_enable_native_pdf_parsing: bool = Field(
        default=True,
        description=(
            "Enable DocAI's native-text-layer extraction for digital/hybrid "
            "PDFs before falling back to OCR. Critical for bilingual docs where "
            "English is digital and Tamil is scanned."
        ),
    )
    google_docai_compute_style_info: bool = Field(
        default=True,
        description=(
            "Extract font, size and style metadata (bold/italic). Enables "
            "accurate heading / paragraph / table discrimination in the parser."
        ),
    )
    google_docai_min_confidence: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Drop any OCR element whose confidence score is below this "
            "threshold. Eliminates low-quality noise blocks from the output."
        ),
    )
    google_docai_max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Maximum retry attempts on transient DocAI errors. "
            "Retries use exponential back-off (1 s, 2 s, 4 s, …)."
        ),
    )

    google_visionai_credentials: str = ""
    google_visionai_language_hints: str = "en,ta"
    google_visionai_timeout: float = Field(default=120.0, ge=5.0, le=3600.0)
    google_visionai_max_retries: int = Field(default=3, ge=1, le=10)

    # ── Google Cloud Translation ─────────────────────────────────────────
    google_translate_credentials: str = Field(
        default="",
        description=(
            "Optional path to a Google Cloud service-account JSON key file for "
            "Cloud Translation. If empty, falls back to DocAI/Vision credentials "
            "or GOOGLE_APPLICATION_CREDENTIALS."
        ),
    )
    google_translate_project_id: str = Field(
        default="",
        description="GCP project ID for Cloud Translation. Falls back to detected credentials project.",
    )
    google_translate_location: str = Field(
        default="global",
        description="Cloud Translation v3 location, usually 'global'.",
    )
    google_translate_timeout: float = Field(default=45.0, ge=5.0, le=300.0)
    google_translate_provider: Literal["free", "cloud", "auto"] = Field(
        default="free",
        description=(
            "Translation provider for PG Processed display translation. "
            "'free' uses Google's public translate web endpoint without credentials; "
            "'cloud' uses official Google Cloud Translation v3; 'auto' tries cloud then free."
        ),
    )

    # ── Native PDF extraction (pdfplumber) — Phase 0b ────────────────────
    # When True and OCR_ENGINE!=paddle, the upload pipeline first runs
    # pdf_detector. If the PDF has a real text layer, pdfplumber extracts
    # structured pages directly — skipping Google DocAI/Vision billing.
    use_native_extractor: bool = Field(
        default=True,
        description=(
            "Enable pdfplumber-based native PDF extraction for digital PDFs. "
            "Saves DocAI/Vision API cost on PDFs that already have a text layer."
        ),
    )
    native_extractor_min_chars_per_page: int = Field(
        default=80,
        ge=10,
        le=10_000,
        description="Threshold to classify a PDF as native (avg chars/page).",
    )

    # ── Remote OCR microservice ──────────────────────────────────────────
    # Used when ``OCR_ENGINE=remote``. Points at an externally-hosted
    # OCR API (Surya, Tesseract, etc.). The container image stays small;
    # the heavy model artefacts live in this service only.
    ocr_remote_url: str = Field(
        default="",
        description="Base URL of the external OCR service (e.g. https://ocr.example.com).",
    )
    ocr_remote_auth: str = Field(
        default="",
        description="Optional HTTP Basic credentials in the form 'user:password'.",
    )
    ocr_remote_api_key: str = Field(
        default="",
        description="Optional bearer token. Sent as 'Authorization: Bearer <key>'.",
    )
    ocr_remote_model: str = Field(
        default="surya",
        description="Model identifier the remote service should use (e.g. surya, tesseract-tam).",
    )
    ocr_remote_timeout_seconds: float = Field(default=120.0, ge=5.0, le=3600.0)

    # ── Outbound HTTP TLS verification ───────────────────────────────────
    # Set ``HTTP_TLS_VERIFY=false`` ONLY in development against self-signed
    # services. Production should always be ``true`` (the default) so that
    # MITM attempts on the OCR / LLM / embedding traffic cannot succeed.
    http_tls_verify: bool = Field(
        default=True,
        description=(
            "Verify TLS certificates on outbound httpx calls "
            "(remote OCR, LLM, embedding). Disable only for dev testing."
        ),
    )

    # ── HuggingFace Inference (chat / LLM) ────────────────────────────────
    hf_inference_api_key: str = ""
    hf_inference_model: str = ""
    hf_inference_base_url: str = "https://router.huggingface.co/v1"

    # ── Custom AI Model URL (chat / LLM) ──────────────────────────────────
    switch_ai: Literal["huggingface", "ai_model", "open_ai"] = "huggingface"
    ai_model_runtime: Literal["remote", "local"] = Field(
        default="remote",
        description=(
            "Selects which OpenAI-compatible ai_model endpoint to use. "
            "'remote' uses AI_MODEL_REMOTE_*; 'local' uses AI_MODEL_LOCAL_*."
        ),
    )
    ai_model_url: str = ""
    ai_model_auth: str = ""
    ai_model_model: str = ""
    ai_model_embedding_model: str = ""
    ai_model_max_context_chars: int = 3000
    ai_model_remote_url: str = ""
    ai_model_remote_auth: str = ""
    ai_model_remote_model: str = ""
    ai_model_remote_max_context_chars: int = 3000
    ai_model_local_url: str = "http://127.0.0.1:8081/v1"
    ai_model_local_auth: str = ""
    ai_model_local_model: str = "Qwen3-14B-Q4_K_M.gguf"
    ai_model_local_max_context_chars: int = 5000

    # ── OpenAI Integration ────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_api_base_openai: str = "https://api.openai.com/v1"
    llm_model_openai: str = "gpt-3.5-turbo"
    openai_ocr_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for the ocr_open_ai engine. Must be a vision model, e.g. gpt-4o.",
    )

    # ── Semantic search ───────────────────────────────────────────────────
    search_chunk_size: int = 800
    search_chunk_overlap: int = 200
    search_index_version: int = 3
    search_embedding_concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Concurrent embedding requests used while building a document search index.",
    )
    search_embedding_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Attempts per chunk/page embedding before marking that chunk embedding_status=failed.",
    )
    search_index_min_success_ratio: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum embedded/saved chunk ratio required for index_health=healthy.",
    )
    search_index_stale_seconds: int = Field(
        default=300,
        ge=30,
        description="How long a zero-chunk indexing job may remain in progress before retrieval may rebuild it on demand.",
    )
    search_index_wait_seconds: int = Field(
        default=45,
        ge=0,
        le=180,
        description=(
            "When a single-doc index is already running, wait this many seconds "
            "for Mongo chunks before falling back to file TF-IDF."
        ),
    )
    search_answer_min_score: float = 0.18
    # When ``+`` is used in a query, all '+' separated terms must occur in
    # a candidate chunk for it to be promoted to the top of the results.
    search_and_separator: str = Field(
        default="+",
        description=(
            "Separator used in the query string to express boolean AND. "
            "E.g. 'market + nmv' requires both 'market' AND 'nmv' to appear."
        ),
    )
    # ── Shared turn pipeline (intent → retrieve → persona → answer) ───
    search_use_turn_pipeline: bool = Field(
        default=True,
        description="Chat /api/semantic-search uses core.turn_pipeline.run_answer_turn.",
    )
    voice_use_turn_pipeline: bool = Field(
        default=True,
        description="Voice engine uses core.turn_pipeline.run_answer_turn.",
    )
    voice_soft_gate_single_doc: bool = Field(
        default=True,
        description="Allow voice single-doc soft-gate when relevant_matches empty.",
    )
    search_refusal_recovery: bool = Field(
        default=True,
        description="Chat: re-ask/extract when LLM falsely refuses despite evidence.",
    )
    search_info_expose_intent: bool = Field(
        default=False,
        description="When true, add turn_intent/persona/domain to search_info (additive).",
    )

    # ── Auth policy ───────────────────────────────────────────────────────
    # /api/auth/register accepts a `role` field. By default we do NOT
    # honour client-supplied roles — every public registration becomes a
    # standard user. Set ``ALLOW_PUBLIC_ADMIN_REGISTRATION=true`` only in
    # closed test environments.
    allow_public_admin_registration: bool = Field(
        default=False,
        description=(
            "When true, the public /api/auth/register endpoint accepts "
            "role='admin'. Always false in production."
        ),
    )

    # ── MongoDB ───────────────────────────────────────────────────────────
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017", validation_alias="mongo_url"
    )
    mongodb_db_name: str = Field(default="pdf_ocr", validation_alias="mongo_db_name")

    # ── JWT ───────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(
        default="super-secret-key", validation_alias="secret_key"
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 180

    # ── Voice Conversation (STT + TTS) ───────────────────────────────────
    # v2 (2026-06, simplified): the production stack has exactly three
    # engines — Whisper (STT) + XTTS-v2 (TTS) + Silero VAD. See
    # VOICE_PLAN.md for the design rationale.
    # Set NEED_VOICE=false in .env to fully disable.
    need_voice: bool = Field(
        default=True,
        description="Whether to preload STT and TTS models on application startup.",
    )

    # ── STT: Faster-Whisper ──────────────────────────────────────────────
    whisper_model_path: str = Field(
        default="",
        description=(
            "Optional filesystem path to a Faster-Whisper model dir (or HF cache "
            "folder). Empty = auto-discover local cache, else use whisper_model_size. "
            "Do not leave a Linux path here on Windows — local fallback will fail."
        ),
    )
    whisper_model_size: str = Field(
        default="large-v3",
        description=(
            "Whisper model size / HF id when path is empty or missing "
            "(tiny|base|small|medium|large-v3)."
        ),
    )
    whisper_device: str = Field(
        default="cpu",
        description="Device for Whisper inference: cpu | cuda. CPU is the supported path.",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="CTranslate2 compute type: int8 (CPU, recommended) | float16 (GPU).",
    )

    # ── TTS: Coqui XTTS-v2 (multilingual + voice cloning) ────────────────
    # XTTS handles Tamil + English + Tanglish + voice cloning in one model.
    xtts_enabled: bool = Field(
        default=True,
        description="Enable Coqui XTTS-v2 for multilingual TTS + voice cloning. ~2 GB.",
    )
    xtts_model_path: str = Field(
        default="/opt/models/voice/xtts-v2",
        description="Path to XTTS-v2 model directory (or HF Hub id if not local).",
    )
    xtts_speaker_dir: str = Field(
        default="/opt/models/voice/xtts-v2/speakers",
        description="Directory holding registered voice-clone reference clips.",
    )
    xtts_default_language: str = Field(
        default="en",
        description="XTTS language tag for synthesis: en | ta | hi | es | fr | de | ...",
    )

    # ── VAD: Silero v4 ───────────────────────────────────────────────────
    silero_vad_enabled: bool = Field(
        default=True,
        description="Enable Silero VAD v4 as a front gate for end-of-speech detection.",
    )
    silero_vad_onnx_path: str = Field(
        default="/opt/models/voice/silero-vad/silero_vad.onnx",
        description="Path to silero_vad.onnx (downloaded by download_voice_models.py).",
    )

    # ── Legacy TTS fallback fields (kept for voice/tts_service.py in-process fallback) ──
    # The main backend still has an in-process TTS fallback for environments
    # without the voice_api microservice. It uses Kokoro (English-only) so
    # Tamil responses require the microservice to be reachable.
    tts_model_path: str = Field(
        default="",
        description="Path to Kokoro TTS model (in-process fallback only; not used by the microservice).",
    )
    tts_voice: str = Field(
        default="af_heart",
        description="Kokoro voice preset name for the in-process fallback.",
    )
    tts_device: str = Field(
        default="cpu",
        description="Device for in-process TTS fallback: cpu | cuda.",
    )
    tts_sample_rate: int = Field(
        default=24000,
        description="In-process TTS fallback sample rate in Hz.",
    )

    # ── Voice session + remote API ───────────────────────────────────────
    voice_max_session_duration: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Maximum voice session duration in seconds (default 30 min).",
    )
    voice_model_url: str = Field(
        default="http://localhost:8003",
        description="URL of the standalone voice API server (microservice).",
    )
    voice_api_key: str = Field(
        default="",
        description="Optional bearer token required for /v1/* routes on the remote voice API.",
    )
    voice_silence_timeout: float = Field(
        default=1.5,
        ge=0.3,
        le=5.0,
        description="Seconds of silence to detect end-of-utterance.",
    )
    voice_max_context_turns: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Number of conversation turns kept in session memory.",
    )

    # ── Embeddings (PROD default: remote all-MiniLM-L6-v2) ────────────────
    # runtime: remote | in_process | auto
    # Prod uses the embedding microservice (MiniLM 384-d). in_process is optional.
    embedding_runtime: str = Field(
        default="remote",
        description="Embedding backend: remote | in_process | auto",
    )
    embedding_local_model_path: str = Field(
        default="",
        description=(
            "Filesystem path or HF id for in-process ST only. "
            "Leave empty in prod when EMBEDDING_RUNTIME=remote."
        ),
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for in-process embeddings: cpu | cuda",
    )
    embedding_preload: bool = Field(
        default=False,
        description="Warm in-process model at startup (no-op for remote runtime).",
    )
    # Remote OpenAI-compatible embedding service
    embedding_model_url: str = ""
    embedding_model_auth: str = ""
    embedding_model_model: str = "bge-m3"
    embedding_model_timeout_seconds: float = Field(default=300.0, ge=10.0, le=1800.0)

    # ── Embedding Caching & Chunk Embeddings (Section 4) ────────────────────
    embedding_cache_enabled: bool = Field(
        default=True,
        description="Cache chunk embeddings in MongoDB to avoid re-embedding identical text across re-indexes.",
    )
    embedding_cache_collection: str = Field(
        default="embedding_cache",
        description="MongoDB collection name for embedding cache.",
    )
    embed_section_chunks: bool = Field(
        default=True,
        description="Generate and index section-level embeddings alongside semantic chunks.",
    )
    embed_parent_chunks: bool = Field(
        default=False,
        description="Generate and index embeddings for parent chunks (default false; parent text fetched by ID).",
    )


    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # ── Convenience helpers ──────────────────────────────────────────────

    def get_max_upload_size_bytes(self) -> int:
        """Maximum accepted upload size, in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Make sure ``upload_dir`` and ``output_dir`` exist on disk."""
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def is_gpu_enabled(self) -> bool:
        """Single source of truth for "should engines try to use the GPU?".

        Honours both the canonical ``GPU_ENABLED`` flag *and* the legacy
        ``USE_GPU`` alias so older deployments don't break.
        """
        return bool(self.gpu_enabled or self.use_gpu)

    def dpi_for_language(self, lang: str | None) -> int:
        """Return the recommended PDF render DPI for an OCR language."""
        if (lang or "").lower().startswith("ta"):
            return self.ocr_dpi_tamil
        return self.ocr_dpi

    def httpx_verify(self) -> bool | str:
        """Return the value to pass to ``httpx.Client(verify=...)``.

        We expose this as a method (not a property) so that callers
        always go through a single, audited code-path. In production
        this returns ``True`` (verify against the system CA bundle).
        """
        return bool(self.http_tls_verify)

    def active_ai_model_url(self) -> str:
        """Resolved OpenAI-compatible LLM endpoint for ``SWITCH_AI=ai_model``."""
        if self.ai_model_runtime == "local":
            return (self.ai_model_local_url or self.ai_model_url).strip()
        return (self.ai_model_remote_url or self.ai_model_url).strip()

    def active_ai_model_auth(self) -> str:
        """Resolved Basic Auth string for ``SWITCH_AI=ai_model``.

        Local llama.cpp servers normally do not need auth, so an empty
        value is valid when ``AI_MODEL_RUNTIME=local``.
        """
        if self.ai_model_runtime == "local":
            return (self.ai_model_local_auth or "").strip()
        return (self.ai_model_remote_auth or self.ai_model_auth).strip()

    def active_ai_model_model(self) -> str:
        """Resolved model id/path for ``SWITCH_AI=ai_model``."""
        if self.ai_model_runtime == "local":
            return (self.ai_model_local_model or self.ai_model_model or "local-model").strip()
        return (self.ai_model_remote_model or self.ai_model_model or "remote-model").strip()

    def active_ai_model_max_context_chars(self) -> int:
        """Resolved context budget for the selected ai_model runtime."""
        if self.ai_model_runtime == "local":
            return int(self.ai_model_local_max_context_chars or self.ai_model_max_context_chars)
        return int(self.ai_model_remote_max_context_chars or self.ai_model_max_context_chars)

    def docai_processor_resource_name(self) -> str:
        """Return the full Document AI processor resource name.

        Priority:
        1. ``GOOGLE_DOCAI_PROCESSOR`` — the full resource name (preferred).
        2. Auto-assembled from ``GOOGLE_DOCAI_PROJECT_ID`` +
           ``GOOGLE_DOCAI_LOCATION`` + ``GOOGLE_DOCAI_PROCESSOR_ID``
           (backward-compat for old split-var configs).

        Raises ``ValueError`` if neither form provides enough information.
        """
        if self.google_docai_processor:
            return self.google_docai_processor

        # Fallback: assemble from split vars
        project = self.google_docai_project_id
        location = self.google_docai_location or "us"
        proc_id = self.google_docai_processor_id

        if project and proc_id:
            return (
                f"projects/{project}/locations/{location}/processors/{proc_id}"
            )

        raise ValueError(
            "Google DocAI processor not configured. Set GOOGLE_DOCAI_PROCESSOR "
            "to the full resource name, e.g.: "
            "projects/my-project/locations/us/processors/abc123"
        )

    @model_validator(mode="after")
    def validate_olm_ocr_path(self) -> "Settings":
        """Ensure OLM_OCR_PATH is provided when using local olmOCR model."""
        if self.ocr_engine == "olm_ocr" and self.olm_ocr_provider == "local":
            if not self.olm_ocr_path:
                raise ValueError(
                    "OLM_OCR_PATH must be set in .env when using OCR_ENGINE=olm_ocr with local provider.\n"
                    "Please download the model using `python download_olmocr_model.py <path>` and then set OLM_OCR_PATH=<path> in your .env file."
                )
        return self


settings = Settings()

# Force loading from .env if present to override system environment variables (like OPENAI_API_KEY="ollama")
try:
    if os.path.exists(".env"):
        from dotenv import dotenv_values
        env_vals = dotenv_values(".env")
        for key, val in env_vals.items():
            key_lower = key.lower()
            if hasattr(settings, key_lower) and val is not None:
                # Typecast to correct pydantic type if needed
                orig_val = getattr(settings, key_lower)
                if isinstance(orig_val, bool):
                    setattr(settings, key_lower, val.lower() in ("true", "1", "yes"))
                elif isinstance(orig_val, int):
                    setattr(settings, key_lower, int(val))
                elif isinstance(orig_val, float):
                    setattr(settings, key_lower, float(val))
                else:
                    setattr(settings, key_lower, val)
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Failed to override settings with .env: %s", e)

settings.ensure_directories()
