"""
Piper TTS engine.

Fast ONNX-based multilingual TTS that runs natively on CPU. This is the
primary TTS engine for the v2 stack (Coqui XTTS was the original
choice, but it's pinned to Python <3.12 and the production venv is 3.12).

Piper ships one voice per language. The engine detects the dominant
script in the input text and picks the voice automatically:

    • Tamil script (Unicode U+0B80–U+0BFF) → ta_IN-vits-medium
    • English / Latin / Tanglish            → en_US-amy-low

For pure Tanglish (Latin-script Tamil phonetics), the English voice
renders it acceptably. For native-Tamil responses, set
``PIPER_DEFAULT_LANGUAGE=ta`` in the env.

Voice catalog (under ``PIPER_MODEL_DIR``):
    en_US-amy-low.onnx      + .json     (English, fast)
    en_US-lessac-low.onnx   + .json     (English alternative)
    ta_IN-vits-medium.onnx  + .json     (Tamil)
    hi_IN-priyamvada-medium.onnx + .json (Hindi, optional)

Reference: https://github.com/rhasspy/piper
"""

from __future__ import annotations

import logging
import os
from typing import Generator, List, Optional

import numpy as np

from .base import TTSChunk, TTSEngine

logger = logging.getLogger(__name__)

# Script ranges for auto-voice selection
_TAMIL_SCRIPT_RE = __import__("re").compile(r"[\u0B80-\u0BFF]")
_DEFAULT_EN_VOICE = "en_US-amy-low"
# Community-maintained Tamil voice from tinisoft/piper-ta_IN-rasa_female-medium
# (Tamil is not in the official rhasspy/piper-voices catalog).
_DEFAULT_TA_VOICE = "ta_IN-rasa_female-medium"


def _detect_dominant_script(text: str) -> str:
    """Return 'ta' if Tamil-script chars dominate, else 'en'."""
    if not text:
        return "en"
    tamil_chars = sum(1 for c in text if _TAMIL_SCRIPT_RE.match(c))
    if tamil_chars > 0:
        return "ta"
    return "en"


class PiperTTSEngine(TTSEngine):
    """Piper ONNX TTS engine — CPU-native, multilingual."""

    name = "piper"
    kind = "tts"

    def __init__(
        self,
        model_dir: str = "/opt/models/voice/piper",
        default_voice_en: str = _DEFAULT_EN_VOICE,
        default_voice_ta: str = _DEFAULT_TA_VOICE,
        default_language: str = "en",
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.default_voice_en = default_voice_en
        self.default_voice_ta = default_voice_ta
        self.default_language = default_language
        # name -> PiperVoice instance
        self._voices: dict = {}

    # ── Voice preference order (highest priority first) ────────────────────
    # Used as a fallback chain when the requested/default voice is missing.
    _VOICE_FALLBACK_ORDER = ["ta_IN-rasa_female-medium", "ta_IN-vits-medium", "en_US-amy-low"]
    _EN_FALLBACK_ORDER = ["en_US-amy-low", "en_US-lessac-low"]

    # ── Capability ─────────────────────────────────────────────────────────
    def supports(self, language: Optional[str]) -> bool:
        return True  # Piper handles both en and ta

    def list_voices(self) -> List[str]:
        voices: List[str] = []
        if os.path.isdir(self.model_dir):
            for fname in sorted(os.listdir(self.model_dir)):
                if fname.endswith(".onnx"):
                    voices.append(os.path.splitext(fname)[0])
        return voices or [self.default_voice_en, self.default_voice_ta]

    def _resolve_voice(self, voice: str, language_hint: str):
        """Pick a loaded voice with fallback. Returns (voice_obj, voice_id)."""
        # 1. Explicit voice id (if loaded)
        if voice and voice not in ("default", "auto"):
            if voice in self._voices:
                return self._voices[voice], voice
            # requested voice not on disk — fall through to language-based fallback
        # 2. Language-based fallback chain
        if language_hint.startswith("ta"):
            for cand in self._VOICE_FALLBACK_ORDER:
                if cand in self._voices:
                    return self._voices[cand], cand
        else:
            for cand in self._EN_FALLBACK_ORDER:
                if cand in self._voices:
                    return self._voices[cand], cand
        # 3. Last resort: any loaded voice
        if self._voices:
            vid = next(iter(self._voices))
            return self._voices[vid], vid
        return None, None

    # ── Load ───────────────────────────────────────────────────────────────
    async def _load_impl(self) -> None:
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts not installed. pip install piper-tts"
            ) from exc

        if not os.path.isdir(self.model_dir):
            logger.warning("Piper model dir %s does not exist; engine will be a no-op", self.model_dir)
            return

        for fname in sorted(os.listdir(self.model_dir)):
            if not fname.endswith(".onnx"):
                continue
            onnx_path = os.path.join(self.model_dir, fname)
            json_path = onnx_path + ".json"
            if not os.path.isfile(json_path):
                logger.warning("Piper voice %s missing .json metadata; skipping", fname)
                continue
            voice_id = os.path.splitext(fname)[0]
            try:
                self._voices[voice_id] = PiperVoice.load(onnx_path, config_path=json_path)
                logger.info("Loaded Piper voice: %s", voice_id)
            except Exception as exc:
                logger.error("Failed to load Piper voice %s: %s", voice_id, exc)

        if not self._voices:
            logger.warning("No Piper voices loaded from %s", self.model_dir)

    # ── Synthesis ──────────────────────────────────────────────────────────
    def synthesise_sync(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        language: Optional[str] = None,
    ) -> Generator[TTSChunk, None, None]:
        if not self._voices:
            # No voices loaded — emit silence so callers get something
            silence = np.zeros(int(22050 * 0.3), dtype=np.float32)
            yield TTSChunk(text=text, sample_rate=22050, audio_float32=silence, engine=self.name)
            return

        # Detect dominant script when caller didn't force a language.
        script_hint = (language or _detect_dominant_script(text) or self.default_language).lower()

        # Use the resolver so we gracefully fall back if the requested/default
        # voice isn't on disk.
        piper_voice, voice_id = self._resolve_voice(voice, script_hint)

        if piper_voice is None:
            silence = np.zeros(int(22050 * 0.3), dtype=np.float32)
            yield TTSChunk(text=text, sample_rate=22050, audio_float32=silence, engine=self.name)
            return

        sample_rate = int(piper_voice.config.sample_rate)

        try:
            # Piper returns AudioChunk objects with .audio_int16_array
            chunks = list(piper_voice.synthesize(text))
            if not chunks:
                return
            int16 = np.concatenate([c.audio_int16_array for c in chunks])
            audio_np = int16.astype(np.float32) / 32767.0
            peak = float(np.abs(audio_np).max()) if len(audio_np) else 0.0
            if peak > 0:
                audio_np = audio_np / peak * 0.9
            yield TTSChunk(
                text=text,
                sample_rate=sample_rate,
                audio_float32=audio_np.astype(np.float32),
                engine=self.name,
            )
        except Exception as exc:
            logger.error("Piper synthesis failed: %s", exc, exc_info=True)
            silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
            yield TTSChunk(text=text[:50], sample_rate=sample_rate, audio_float32=silence, engine=self.name)
