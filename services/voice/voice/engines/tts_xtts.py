"""
Coqui XTTS-v2 TTS engine.

Multilingual TTS (16 languages incl. Tamil-script support via transliteration
to Latin) with 6-second voice cloning from a reference clip.

Reference: https://huggingface.co/coqui/XTTS-v2
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, Generator, List, Optional

import numpy as np
import soundfile as sf

from .base import TTSChunk, TTSEngine

logger = logging.getLogger(__name__)

XTTS_SAMPLE_RATE = 24_000

# Languages XTTS-v2 handles natively. Tamil is supported via the
# ``ta`` token (the model transliterates internally).
XTTS_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu",
    "hi", "ta",
]

DEFAULT_VOICE_SAMPLES: Dict[str, str] = {}  # populated by config; engine uses
# a built-in "default" speaker embedding if no reference clip is registered.


class XTTSEngine(TTSEngine):
    """XTTS-v2 via the Coqui TTS package."""

    name = "xtts"
    kind = "tts"
    sample_rate = XTTS_SAMPLE_RATE

    def __init__(
        self,
        model_path: str = "coqui/XTTS-v2",
        default_language: str = "en",
        speaker_wav_dir: str = "/models/xtts-v2/speakers",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.default_language = default_language
        self.speaker_wav_dir = speaker_wav_dir
        self._tts: object = None

    # ── Capability ─────────────────────────────────────────────────────────
    def supports(self, language: Optional[str]) -> bool:
        if language in (None, "", "auto"):
            return True
        return language in XTTS_LANGUAGES

    def list_voices(self) -> List[str]:
        # Catalog = registered clone names + the special "default" speaker.
        names: List[str] = ["default"]
        if os.path.isdir(self.speaker_wav_dir):
            for fname in sorted(os.listdir(self.speaker_wav_dir)):
                if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    names.append(f"clone:{os.path.splitext(fname)[0]}")
        return names

    # ── Load ───────────────────────────────────────────────────────────────
    async def _load_impl(self) -> None:
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS not installed. pip install TTS>=0.22"
            ) from exc

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading XTTS-v2 from %s on %s", self.model_path, device)

        self._tts = CoquiTTS(
            model_name=self.model_path,
            progress_bar=False,
            gpu=(device == "cuda"),
        ).to(device)

    # ── Synthesis ──────────────────────────────────────────────────────────
    def synthesise_sync(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        language: Optional[str] = None,
    ) -> Generator[TTSChunk, None, None]:
        if self._tts is None:
            silence = np.zeros(int(XTTS_SAMPLE_RATE * 0.5), dtype=np.float32)
            yield TTSChunk(text=text, sample_rate=XTTS_SAMPLE_RATE, audio_float32=silence, engine=self.name)
            return

        lang = language or self._detect_language(text) or self.default_language
        if lang not in XTTS_LANGUAGES:
            lang = "en"

        speaker_wav = self._resolve_speaker_wav(voice)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                # XTTS writes a single WAV file per call. We split into
                # sentence chunks upstream (in the router) so XTTS still
                # gets the right granularity via repeated calls.
                self._tts.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speaker_wav=speaker_wav,
                    language=lang,
                    speed=speed,
                )
                audio_np, sr = sf.read(tmp_path, dtype="float32")
                if sr != XTTS_SAMPLE_RATE:
                    # Simple linear resample if needed (rare).
                    audio_np = self._linear_resample(audio_np, sr, XTTS_SAMPLE_RATE)
                peak = float(np.abs(audio_np).max()) if len(audio_np) else 0.0
                if peak > 0:
                    audio_np = audio_np / peak * 0.9
                yield TTSChunk(
                    text=text,
                    sample_rate=XTTS_SAMPLE_RATE,
                    audio_float32=audio_np.astype(np.float32),
                    engine=self.name,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            logger.error("XTTS synthesis failed: %s", exc, exc_info=True)
            silence = np.zeros(int(XTTS_SAMPLE_RATE * 0.3), dtype=np.float32)
            yield TTSChunk(text=text[:50], sample_rate=XTTS_SAMPLE_RATE, audio_float32=silence, engine=self.name)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _resolve_speaker_wav(self, voice: str) -> Optional[str]:
        if voice in (None, "", "default"):
            return None
        if voice.startswith("clone:"):
            name = voice[len("clone:"):]
            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                candidate = os.path.join(self.speaker_wav_dir, name + ext)
                if os.path.isfile(candidate):
                    return candidate
            return None
        # Treat as a direct filename
        if os.path.isfile(voice):
            return voice
        return None

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        if not text:
            return None
        if any("\u0B80" <= c <= "\u0BFF" for c in text):  # Tamil
            return "ta"
        if any("\u0900" <= c <= "\u097F" for c in text):  # Devanagari
            return "hi"
        return None

    @staticmethod
    def _linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or len(audio) == 0:
            return audio
        duration = len(audio) / src_sr
        new_len = int(duration * dst_sr)
        xp = np.linspace(0, duration, num=len(audio), endpoint=True)
        xq = np.linspace(0, duration, num=new_len, endpoint=True)
        return np.interp(xq, xp, audio).astype(np.float32)
