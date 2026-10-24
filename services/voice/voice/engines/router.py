"""
Language-aware engine router.

The v2 stack has exactly one engine per role:
    STT → Whisper (handles ta / en / Tanglish code-switching)
    TTS → Piper ONNX (handles en / Tamil-script voice selection)
    VAD → Silero v4 (language-agnostic)

So in practice ``pick_stt`` / ``pick_tts`` always return the single loaded
engine for that role. The framework is kept so that swapping in a
future engine (e.g. F5-TTS) is a one-file change rather than a refactor
of the API layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .base import STTEngine, TTSEngine, VoiceEngine

logger = logging.getLogger(__name__)


# ── Script detection (kept for future multilingual TTS routing) ────────────
# Cheap regex pre-check. Piper uses this kind of script hint to pick an
# installed voice, and the helper remains useful for future routing.

_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


@dataclass
class LanguageHint:
    """Result of cheap script detection."""

    code: str  # "en", "ta", "hi", "zh", ...
    confidence: float  # 0..1, proportion of chars that triggered the script
    needs_multilingual_tts: bool


def detect_text_language(text: str) -> LanguageHint:
    """Cheap O(n) script-based language detection for TTS routing."""
    if not text or not text.strip():
        return LanguageHint("en", 1.0, False)

    total = max(len(text), 1)
    if _TAMIL_RE.search(text):
        return LanguageHint("ta", _script_ratio(text, _TAMIL_RE, total), True)
    if _DEVANAGARI_RE.search(text):
        return LanguageHint("hi", _script_ratio(text, _DEVANAGARI_RE, total), True)
    if _BENGALI_RE.search(text):
        return LanguageHint("bn", _script_ratio(text, _BENGALI_RE, total), True)
    if _TELUGU_RE.search(text):
        return LanguageHint("te", _script_ratio(text, _TELUGU_RE, total), True)
    if _THAI_RE.search(text):
        return LanguageHint("th", _script_ratio(text, _THAI_RE, total), True)
    if _ARABIC_RE.search(text):
        return LanguageHint("ar", _script_ratio(text, _ARABIC_RE, total), True)
    if _CJK_RE.search(text):
        return LanguageHint("zh", _script_ratio(text, _CJK_RE, total), True)
    if _CYRILLIC_RE.search(text):
        return LanguageHint("ru", _script_ratio(text, _CYRILLIC_RE, total), True)
    return LanguageHint("en", 1.0, False)


def _script_ratio(text: str, pattern: re.Pattern, total: int) -> float:
    hits = sum(1 for ch in text if pattern.match(ch))
    return hits / total


# ── Engine registry ─────────────────────────────────────────────────────────

class EngineRegistry:
    """Holds the loaded engines and answers routing queries."""

    def __init__(self) -> None:
        self._stt: List[STTEngine] = []
        self._tts: List[TTSEngine] = []

    # ── Registration ───────────────────────────────────────────────────────
    def register_stt(self, engine: STTEngine) -> None:
        self._stt.append(engine)

    def register_tts(self, engine: TTSEngine) -> None:
        self._tts.append(engine)

    # ── Accessors ──────────────────────────────────────────────────────────
    @property
    def stt_engines(self) -> List[STTEngine]:
        return [e for e in self._stt if e.is_loaded]

    @property
    def tts_engines(self) -> List[TTSEngine]:
        return [e for e in self._tts if e.is_loaded]

    @property
    def all_engines(self) -> List[VoiceEngine]:
        # Stable display order: STT first, TTS second
        return list(self._stt) + list(self._tts)

    # ── Routing ────────────────────────────────────────────────────────────
    def pick_stt(self, language: Optional[str]) -> Optional[STTEngine]:
        """Return the STT engine (only one loaded — Whisper).

        Returns ``None`` when nothing is loaded yet, so the caller can
        return a 503 instead of crashing. ``language`` is accepted for
        API compatibility but ignored for routing decisions.
        """
        available = self.stt_engines
        if not available:
            return None
        for preferred in ("whisper",):
            for engine in available:
                if engine.name == preferred:
                    return engine
        return available[0]

    def pick_tts(self, text: str, voice: str = "default") -> Optional[TTSEngine]:
        """Return the TTS engine (only one loaded — Piper).

        Piper handles synthesis while the engine itself picks a voice from
        the requested/default voice and text language hint.
        """
        available = self.tts_engines
        if not available:
            return None
        for preferred in ("piper",):
            for engine in available:
                if engine.name == preferred:
                    return engine
        return available[0]

    def list_voices(self) -> List[dict]:
        """Aggregate TTS voice catalog across all loaded engines."""
        catalog: List[dict] = []
        for engine in self.tts_engines:
            for voice in engine.list_voices():
                catalog.append({"engine": engine.name, "voice": voice})
        return catalog

    def snapshot(self) -> List[dict]:
        """Serialisable engine map for /health and /v1/engines."""
        return [engine.status() for engine in self.all_engines]


# ── Module-level singleton ──────────────────────────────────────────────────
_registry: Optional[EngineRegistry] = None


def get_registry() -> EngineRegistry:
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry


def reset_registry() -> None:
    """Test helper: drop the registry so a fresh one is built."""
    global _registry
    _registry = None
