"""
Text-to-Speech service using Kokoro TTS.

Provides streaming sentence-by-sentence audio generation for low-latency
voice responses. Each sentence is yielded as a PCM audio chunk so the
frontend can begin playback while subsequent sentences are still being
generated.

Falls back to a silent placeholder if the TTS model fails to load,
ensuring the system never crashes due to TTS issues.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Generator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────
_pipeline = None
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")
_available = None

TTS_SAMPLE_RATE = 24_000  # Kokoro native output rate


def _load_pipeline():
    """Load the Kokoro TTS pipeline (called once)."""
    global _pipeline, _available

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.warning(
            "kokoro package not installed. TTS will be disabled. "
            "Install with: pip install kokoro misaki[en]"
        )
        _available = False
        return None

    from config import settings

    lang_code = "a"  # American English (best for Tanglish/mixed content)
    logger.info("Loading Kokoro TTS pipeline (lang_code=%s)...", lang_code)

    try:
        _pipeline = KPipeline(lang_code=lang_code)
        _available = True
        logger.info("Kokoro TTS pipeline loaded successfully!")
    except Exception as e:
        logger.error("Failed to load Kokoro TTS: %s", e, exc_info=True)
        _available = False
        _pipeline = None

    return _pipeline


def get_tts_pipeline():
    """Get or load the singleton TTS pipeline."""
    global _pipeline
    if _pipeline is None and _available is not False:
        _load_pipeline()
    return _pipeline


def is_tts_available() -> bool:
    """Check if TTS is loaded and ready."""
    return _available and _pipeline is not None


# ── Sentence splitting ───────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(
    r'(?<=[.!?।])\s+|(?<=[.!?।])$|(?<=\n)\s*',
    re.UNICODE,
)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental TTS.

    Aims for natural speech pauses. Handles English periods/question marks
    and Tamil purna viram (।).
    """
    if not text or not text.strip():
        return []

    # First try splitting on sentence boundaries
    parts = _SENTENCE_RE.split(text.strip())
    sentences = [s.strip() for s in parts if s and s.strip()]

    if not sentences:
        return [text.strip()]

    # Merge very short fragments into the previous sentence
    merged = []
    for s in sentences:
        if merged and len(s) < 15 and not s.endswith((".", "!", "?", "।")):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    return merged


# ── Synchronous TTS generation ───────────────────────────────────────────────

def _generate_sync(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    cancel_event: Optional[asyncio.Event] = None,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """Generate speech for text, yielding (sentence_text, audio) pairs.

    Each yield produces one sentence worth of audio as a float32 numpy
    array at 24 kHz. The caller can start playback immediately while
    subsequent sentences are still being synthesised.

    Args:
        text: Input text to speak.
        voice: Kokoro voice preset name.
        speed: Speech speed multiplier (1.0 = normal).
        cancel_event: If set, generation stops between sentences.

    Yields:
        (sentence_text, audio_float32_24khz)
    """
    pipeline = get_tts_pipeline()
    if pipeline is None:
        logger.warning("TTS not available — generating silence")
        # Yield 0.5s of silence as placeholder
        silence = np.zeros(int(TTS_SAMPLE_RATE * 0.5), dtype=np.float32)
        yield (text, silence)
        return

    try:
        generator = pipeline(text, voice=voice, speed=speed)

        for graphemes, phonemes, audio_tensor in generator:
            # Check for cancellation between sentences
            if cancel_event is not None and cancel_event.is_set():
                logger.info("TTS generation cancelled by interrupt")
                return

            if audio_tensor is not None:
                audio_np = audio_tensor.cpu().numpy() if hasattr(audio_tensor, 'cpu') else np.array(audio_tensor, dtype=np.float32)

                # Normalize audio to prevent clipping
                max_val = np.abs(audio_np).max()
                if max_val > 0:
                    audio_np = audio_np / max_val * 0.9

                sentence_text = graphemes or ""
                yield (sentence_text, audio_np.astype(np.float32))

    except Exception as e:
        logger.error("TTS generation failed: %s", e, exc_info=True)
        # Yield silence so the conversation flow continues
        silence = np.zeros(int(TTS_SAMPLE_RATE * 0.3), dtype=np.float32)
        yield (text[:50], silence)


# ── Async wrapper ────────────────────────────────────────────────────────────

async def generate_speech(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[Tuple[str, bytes], None]:
    """Async generator that yields (sentence_text, pcm_int16_bytes) pairs.

    Each yielded chunk is a complete sentence of audio encoded as
    16-bit signed PCM at 24 kHz — ready to send over WebSocket.

    Args:
        text: Text to convert to speech.
        voice: Voice preset (default from settings).
        speed: Speech speed multiplier.
        cancel_event: Set this event to cancel mid-generation.

    Yields:
        (sentence_text, pcm_bytes) — pcm_bytes is Int16 PCM at 24 kHz.
    """
    from config import settings
    from voice.audio_processor import float32_to_int16_bytes
    import httpx
    import json
    import base64

    voice = voice or getattr(settings, "tts_voice", "af_heart")
    voice_url = getattr(settings, "voice_model_url", "").strip()
    voice_api_key = getattr(settings, "voice_api_key", "").strip()

    if voice_url:
        url = f"{voice_url.rstrip('/')}/tts"
        payload = {"text": text, "voice": voice, "speed": speed}
        headers = {}
        if voice_api_key:
            headers["Authorization"] = f"Bearer {voice_api_key}"
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    chunks_seen = 0
                    async for line in response.aiter_lines():
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info("Remote TTS streaming cancelled by interrupt")
                            return
                        if not line or not line.strip():
                            continue
                        data = json.loads(line)
                        sentence_text = data["text"]
                        pcm_bytes = base64.b64decode(data["audio_b64"])
                        remote_sr = data.get("sample_rate", 22050)
                        
                        if remote_sr != 24000:
                            from voice.audio_processor import int16_bytes_to_float32, float32_to_int16_bytes, resample
                            audio_f32 = int16_bytes_to_float32(pcm_bytes)
                            audio_f32 = resample(audio_f32, remote_sr, 24000)
                            pcm_bytes = float32_to_int16_bytes(audio_f32)
                        chunks_seen += 1
                        yield sentence_text, pcm_bytes
                    if chunks_seen == 0:
                        logger.warning("Remote TTS returned 200 but produced no audio chunks: %s", url)
            return
        except Exception as exc:
            logger.error("Remote TTS streaming failed, falling back to local: %s", exc)

    loop = asyncio.get_running_loop()

    # Run the synchronous generator in a thread
    # We collect one sentence at a time to yield asynchronously
    def _gen_one_sentence(gen_state: dict):
        """Pull the next sentence from the generator."""
        gen = gen_state.get("gen")
        if gen is None:
            gen = _generate_sync(text, voice=voice, speed=speed, cancel_event=cancel_event)
            gen_state["gen"] = gen

        try:
            return next(gen)
        except StopIteration:
            return None

    gen_state: dict = {}

    while True:
        if cancel_event is not None and cancel_event.is_set():
            return

        result = await loop.run_in_executor(
            _executor,
            _gen_one_sentence,
            gen_state,
        )

        if result is None:
            break

        sentence_text, audio_float32 = result
        pcm_bytes = float32_to_int16_bytes(audio_float32)
        yield (sentence_text, pcm_bytes)


async def preload_model() -> None:
    """Pre-load the TTS model locally or verify remote API connection."""
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
    await loop.run_in_executor(_executor, get_tts_pipeline)
    if _available:
        logger.info("TTS model pre-loaded.")
    else:
        logger.warning("TTS model not available — voice responses will be text-only.")
