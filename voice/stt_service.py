"""
Speech-to-Text service using Faster-Whisper (CTranslate2).

Loads the full Whisper Large v3 model from a local path and provides
synchronous transcription wrapped in async for the FastAPI event loop.

Model path resolution:
    • Supports HuggingFace cache layout  (models--Systran--faster-whisper-large-v3/snapshots/...)
    • Supports direct path to model directory (containing model.bin)
    • Falls back to model size string for auto-download
"""

from __future__ import annotations

import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton model ──────────────────────────────────────────────────────────
_model = None
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")

# A short Tamil seed prompt biases the decoder toward Tamil script (rather than
# transliteration) when transcribing in Tamil.
_TA_INITIAL_PROMPT = "இது ஒரு தமிழ் உரையாடல்."

# Guards against Whisper's classic failure modes on short/noisy mic audio:
# runaway repetition (compression-ratio + repetition penalties), low-confidence
# hallucinations (log-prob threshold), and silence read as speech. Mirrors the
# standalone Voice API engine so local-fallback behaviour matches the primary.
_ANTI_HALLUCINATION_PARAMS = dict(
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
)


def _detect_language_restricted(model, audio) -> Optional[str]:
    """Detect language but restrict the answer to {en, ta}.

    Returns ``"en"``/``"ta"`` (higher-scoring of the two) or ``None`` when the
    faster-whisper build lacks ``detect_language`` (caller then falls back to
    blind auto-detect + force-Tamil).
    """
    detect = getattr(model, "detect_language", None)
    if detect is None:
        return None
    try:
        result = detect(audio)
    except Exception as exc:
        logger.debug("detect_language unavailable/failed: %s", exc)
        return None

    all_probs = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
    if not all_probs:
        return None

    probs = {}
    for item in all_probs:
        if isinstance(item, dict):
            probs[item.get("language")] = float(item.get("probability", 0.0) or 0.0)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            probs[item[0]] = float(item[1] or 0.0)

    p_en, p_ta = probs.get("en", 0.0), probs.get("ta", 0.0)
    if p_en <= 0.0 and p_ta <= 0.0:
        return None
    chosen = "ta" if p_ta >= p_en else "en"
    logger.info(
        "Restricted language detection → %s (p_en=%.2f, p_ta=%.2f)", chosen, p_en, p_ta
    )
    return chosen


def _resolve_model_path(path: str) -> str:
    """Resolve a HuggingFace Hub cache directory to the actual model folder.

    HF cache layout:
        models--Systran--faster-whisper-large-v3/
            blobs/
            refs/
            snapshots/
                <hash>/
                    config.json
                    model.bin
                    ...

    If ``path`` is already a direct model directory (contains model.bin),
    it is returned as-is.
    """
    # Already a direct model directory?
    if os.path.isfile(os.path.join(path, "model.bin")):
        return path

    # HF cache format — resolve snapshot
    snapshots_dir = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = sorted(os.listdir(snapshots_dir))
        if snapshots:
            resolved = os.path.join(snapshots_dir, snapshots[-1])
            if os.path.isfile(os.path.join(resolved, "model.bin")):
                logger.info("Resolved HF cache snapshot: %s", resolved)
                return resolved

    # Return as-is and let faster-whisper handle it (e.g. model size string)
    return path


def _looks_like_filesystem_path(value: str) -> bool:
    """True for absolute/relative paths that must exist on disk (not HF repo ids)."""
    if not value:
        return False
    # Windows drive / UNC, Unix absolute, or explicit relative segments
    if value.startswith(("/", "\\")) or (len(value) >= 2 and value[1] == ":"):
        return True
    if "\\" in value or value.startswith((".", "..")):
        return True
    return False


def _is_usable_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "model.bin")):
        return True
    snapshots = os.path.join(path, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    for name in os.listdir(snapshots):
        if os.path.isfile(os.path.join(snapshots, name, "model.bin")):
            return True
    return False


def _discover_local_whisper(size: str) -> Optional[str]:
    """Search common Windows/Linux cache locations for a cached model."""
    size = (size or "large-v3").strip()
    # Map size → HF cache folder name fragment
    folder = f"models--Systran--faster-whisper-{size}"
    home = os.path.expanduser("~")
    candidates = [
        os.path.join("D:\\models", folder),
        os.path.join(home, ".cache", "huggingface", "hub", folder),
        os.path.join("C:\\models", folder),
        os.path.join("/models", folder),
        os.environ.get("LOCAL_WHISPER_MODEL_PATH", ""),
    ]
    for raw in candidates:
        if not raw:
            continue
        if _is_usable_model_dir(raw):
            return _resolve_model_path(raw)
    return None


def _pick_whisper_source() -> str:
    """Choose a model path or size string that faster-whisper can load.

    Production Linux paths in config must not break Windows local dev when the
    remote voice API is down (502). Prefer existing local dirs, then HF size id.
    """
    from config import settings

    configured = (getattr(settings, "whisper_model_path", None) or "").strip()
    size = (getattr(settings, "whisper_model_size", None) or "large-v3").strip()

    if configured:
        if _is_usable_model_dir(configured):
            return _resolve_model_path(configured)
        if _looks_like_filesystem_path(configured):
            logger.warning(
                "WHISPER_MODEL_PATH=%r is missing on this machine "
                "(common when a Linux path is left in .env on Windows). "
                "Falling back to local cache / model size %r.",
                configured,
                size,
            )
        else:
            # Treat as HF repo id / size string
            return configured

    discovered = _discover_local_whisper(size)
    if discovered:
        logger.info("Using discovered local Whisper model: %s", discovered)
        return discovered

    # faster-whisper accepts sizes: tiny, base, small, medium, large-v3, ...
    logger.info("Using Faster-Whisper model size id: %s", size)
    return size


def _load_model():
    """Load the Whisper model (called once, then cached)."""
    global _model

    from config import settings

    device = settings.whisper_device
    compute_type = settings.whisper_compute_type

    resolved_path = _pick_whisper_source()

    # Auto-detect device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # On CPU, auto → int8 (fastest); on CUDA, auto → float16
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"

    logger.info(
        "Loading Faster-Whisper model: path=%s  device=%s  compute_type=%s",
        resolved_path, device, compute_type,
    )

    from faster_whisper import WhisperModel

    _model = WhisperModel(
        resolved_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=min(os.cpu_count() or 4, 8),  # Use up to 8 threads
    )

    # Warm up with a tiny silent audio
    warmup_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
    list(_model.transcribe(warmup_audio, language="en")[0])

    logger.info("Whisper model loaded and warmed up successfully!")
    return _model


def get_stt_model():
    """Get or load the singleton Whisper model."""
    global _model
    if _model is None:
        _load_model()
    return _model


# ── Synchronous transcription ────────────────────────────────────────────────

def _transcribe_sync(
    audio: np.ndarray,
    language: Optional[str] = None,
) -> Tuple[str, str, List[dict]]:
    """Transcribe a float32 audio array (16 kHz mono).

    Args:
        audio: float32 numpy array, values in [-1, 1], 16 kHz.
        language: Force language code (e.g. "ta", "en"). None = auto-detect.

    Returns:
        (full_text, detected_language, segments)
        where each segment is {"start": float, "end": float, "text": str}.
    """
    from voice.audio_processor import WHISPER_SAMPLE_RATE, compute_db

    model = get_stt_model()
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    duration_s = len(audio) / WHISPER_SAMPLE_RATE if len(audio) else 0.0
    db = compute_db(audio)

    if duration_s < 0.25:
        return "", language or "en", []

    if db < -42.0:
        logger.info(
            "STT skipped quiet audio: duration=%.2fs level=%.1fdB",
            duration_s,
            db,
        )
        return "", language or "en", []

    def _run_whisper(lang: Optional[str], use_vad: bool, initial_prompt: Optional[str] = None):
        return model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            best_of=5,
            patience=1.0,
            vad_filter=use_vad,
            vad_parameters=dict(
                min_silence_duration_ms=700,
                speech_pad_ms=300,
            ),
            condition_on_previous_text=False,
            suppress_blank=True,
            initial_prompt=initial_prompt,
            **_ANTI_HALLUCINATION_PARAMS,
        )

    use_vad = duration_s >= 1.0

    # Restrict auto-detection to {en, ta} up front (see the standalone Voice API
    # engine for the rationale) rather than transcribing blind and re-running a
    # second full pass to correct a non-bilingual mis-detection.
    forced = language is not None
    decode_lang = language if forced else _detect_language_restricted(model, audio)
    initial_prompt = _TA_INITIAL_PROMPT if decode_lang == "ta" else None

    segments_gen, info = _run_whisper(decode_lang, use_vad, initial_prompt)
    segments_list = list(segments_gen)

    detected_lang = decode_lang or info.language or "en"

    if not segments_list and use_vad and db > -34.0:
        logger.info(
            "Whisper VAD removed all audio despite level %.1fdB; retrying once without VAD",
            db,
        )
        segments_gen, info = _run_whisper(decode_lang, False, initial_prompt)
        segments_list = list(segments_gen)
        detected_lang = decode_lang or info.language or detected_lang

    if not segments_list:
        logger.info(
            "STT produced no speech segments: lang=%s duration=%.2fs level=%.1fdB prob=%.2f",
            detected_lang,
            duration_s,
            db,
            info.language_probability,
        )
        return "", detected_lang, []

    # Legacy safety net for faster-whisper builds without detect_language: if
    # blind auto-detect returned a non-bilingual language, force Tamil ('ta')
    # to guarantee correct native/phonetic output instead of gibberish.
    if not forced and decode_lang is None and detected_lang not in {"en", "ta"}:
        logger.info(
            "Auto-detect returned non-bilingual language '%s' (prob %.2f) — forcing Tamil ('ta') transcription",
            detected_lang, info.language_probability
        )
        segments_gen, info = _run_whisper("ta", use_vad, _TA_INITIAL_PROMPT)
        segments_list = list(segments_gen)
        if not segments_list and use_vad and db > -34.0:
            logger.info(
                "Forced Tamil pass produced no VAD segments; retrying Tamil once without VAD"
            )
            segments_gen, info = _run_whisper("ta", False, _TA_INITIAL_PROMPT)
            segments_list = list(segments_gen)

    segments = []
    full_text_parts = []
    for seg in segments_list:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
            })
            full_text_parts.append(text)

    full_text = " ".join(full_text_parts)
    detected_lang = info.language or "en"

    logger.info(
        "STT result: lang=%s  text_len=%d  segments=%d  duration=%.1fs  prob=%.2f",
        detected_lang, len(full_text), len(segments),
        info.duration, info.language_probability,
    )

    return full_text, detected_lang, segments


# ── Async wrapper ────────────────────────────────────────────────────────────

async def transcribe(
    audio: np.ndarray,
    language: Optional[str] = None,
) -> Tuple[str, str, List[dict]]:
    """Async transcription — routes to the standalone Voice API or runs Whisper locally.

    Args:
        audio: float32 numpy array, values in [-1, 1], 16 kHz.
        language: Force language code or None for auto-detect.

    Returns:
        (full_text, detected_language, segments)
    """
    from config import settings
    import httpx

    import time as _time
    from voice.audio_processor import WHISPER_SAMPLE_RATE as _stt_sr

    try:
        audio_ms = int(len(audio) / max(_stt_sr, 1) * 1000)
    except Exception:
        audio_ms = 0

    t0 = _time.perf_counter()
    voice_url = getattr(settings, "voice_model_url", "").strip()
    if voice_url:
        from voice.audio_processor import float32_to_int16_bytes
        pcm_bytes = float32_to_int16_bytes(audio)
        voice_api_key = getattr(settings, "voice_api_key", "").strip()
        headers = {}
        if voice_api_key:
            headers["Authorization"] = f"Bearer {voice_api_key}"
        
        url = f"{voice_url.rstrip('/')}/transcribe"
        params = {}
        if language:
            params["language"] = language

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    url,
                    content=pcm_bytes,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                stt_ms = (_time.perf_counter() - t0) * 1000
                rtf = (stt_ms / audio_ms) if audio_ms > 0 else 0.0
                logger.info(
                    "stage=stt backend=remote audio_ms=%d stt_ms=%.0f rtf=%.2f",
                    audio_ms,
                    stt_ms,
                    rtf,
                )
                return data["text"], data["language"], data["segments"]
        except Exception as exc:
            logger.error("Remote STT request failed, falling back to local: %s", exc)
            t0 = _time.perf_counter()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        _transcribe_sync,
        audio,
        language,
    )
    stt_ms = (_time.perf_counter() - t0) * 1000
    rtf = (stt_ms / audio_ms) if audio_ms > 0 else 0.0
    logger.info(
        "stage=stt backend=local audio_ms=%d stt_ms=%.0f rtf=%.2f",
        audio_ms,
        stt_ms,
        rtf,
    )
    return result


async def preload_model() -> None:
    """Pre-load the Whisper model locally or verify remote API connection."""
    from config import settings
    import httpx

    voice_url = getattr(settings, "voice_model_url", "").strip()
    if voice_url:
        url = f"{voice_url.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    logger.info("Remote Voice API Server is online and verified.")
                    return
        except Exception as exc:
            logger.warning("Remote Voice API Server health check failed: %s", exc)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, get_stt_model)
    logger.info("STT model pre-loaded.")
