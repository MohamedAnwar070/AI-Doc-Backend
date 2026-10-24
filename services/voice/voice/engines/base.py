"""
Voice Engine Base Classes.

Defines the abstract contracts that every STT, TTS, and VAD engine must
satisfy so the voice_api.py layer can talk to any combination of
backends uniformly.

Engine lifecycle:
    1. Engine object is constructed (no model loaded yet, no GPU locked).
    2. ``async load()`` is awaited on app startup — this is where heavy
       I/O (downloading weights, allocating GPU memory) happens.
    3. Sync inference methods (transcribe / synthesise / vad) are run
       inside the FastAPI threadpool because they release the GIL for
       native C/C++/CUDA kernels.
    4. ``async unload()`` releases memory on shutdown.

Every engine also exposes ``supports(language)`` so the router can pick
the right backend without ever importing engine-specific code.
"""

from __future__ import annotations

import abc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Result dataclasses ──────────────────────────────────────────────────────

@dataclass
class STTResult:
    """One full transcription pass."""

    text: str
    language: str
    language_probability: float = 1.0
    duration: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    engine: str = ""
    latency_ms: float = 0.0


@dataclass
class TTSChunk:
    """One sentence worth of synthesised audio."""

    text: str
    sample_rate: int
    audio_float32: np.ndarray
    engine: str = ""

    def to_int16_bytes(self) -> bytes:
        """Convert float32 [-1,1] to raw Int16 PCM bytes."""
        clipped = np.clip(self.audio_float32, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()


@dataclass
class VADSegment:
    """A speech region detected by VAD."""

    start: float
    end: float


# ── Base engine ─────────────────────────────────────────────────────────────

class VoiceEngine(abc.ABC):
    """Common contract for every voice engine."""

    name: str = "base"
    kind: str = "stt"  # "stt" | "tts" | "vad"

    def __init__(self) -> None:
        self._loaded = False
        self._load_error: Optional[str] = None

    # ── State ───────────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def status(self) -> Dict[str, Any]:
        """Serialisable health snapshot for /health and /v1/engines."""
        return {
            "name": self.name,
            "kind": self.kind,
            "loaded": self._loaded,
            "error": self._load_error,
        }

    # ── Lifecycle ───────────────────────────────────────────────────────────
    async def load(self) -> None:
        """Load weights / pipeline. Idempotent."""
        if self._loaded:
            return
        t0 = time.time()
        try:
            await self._load_impl()
            self._loaded = True
            dt = (time.time() - t0) * 1000
            logger.info("Engine %s loaded in %.0f ms", self.name, dt)
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Engine %s failed to load: %s", self.name, exc, exc_info=True)

    @abc.abstractmethod
    async def _load_impl(self) -> None: ...

    async def unload(self) -> None:
        self._loaded = False

    # ── Capability query ───────────────────────────────────────────────────
    def supports(self, language: Optional[str]) -> bool:
        """Return True if the engine can handle ``language`` (None = wildcard)."""
        return True


# ── STT engine ──────────────────────────────────────────────────────────────

class STTEngine(VoiceEngine):
    """Abstract speech-to-text engine."""

    kind = "stt"

    @abc.abstractmethod
    def transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> STTResult:
        """Blocking transcription. Must be safe to run inside a threadpool."""
        ...

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> STTResult:
        """Async wrapper that offloads ``transcribe_sync`` to the default executor."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.transcribe_sync, audio, sample_rate, language
        )


# ── TTS engine ──────────────────────────────────────────────────────────────

class TTSEngine(VoiceEngine):
    """Abstract text-to-speech engine."""

    kind = "tts"

    @abc.abstractmethod
    def synthesise_sync(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        language: Optional[str] = None,
    ) -> Generator[TTSChunk, None, None]:
        """Blocking sentence-streaming synthesis.

        Each yield is one sentence worth of audio so the caller can begin
        playback immediately. Must release the GIL during the heavy kernel.
        """
        ...

    async def synthesise(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        language: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """Async sentence-streaming wrapper.

        Yields one TTSChunk per sentence. The next sentence is only pulled
        from the sync generator when the consumer awaits it, which is what
        lets us cancel mid-stream on barge-in.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        gen = self.synthesise_sync(text, voice=voice, speed=speed, language=language)
        while True:
            chunk = await loop.run_in_executor(None, _next_or_none, gen)
            if chunk is None:
                return
            yield chunk

    def list_voices(self) -> List[str]:
        """Return the catalog of voices this engine exposes."""
        return ["default"]


# ── VAD engine ──────────────────────────────────────────────────────────────

class VADEngine(VoiceEngine):
    """Abstract voice-activity-detection engine."""

    kind = "vad"

    @abc.abstractmethod
    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        """Return True if the chunk contains speech."""
        ...

    def segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        min_silence_ms: int = 700,
        speech_pad_ms: int = 300,
    ) -> List[VADSegment]:
        """Split ``audio`` into speech segments using the rolling VAD."""
        ...


# ── Helpers ─────────────────────────────────────────────────────────────────

def _next_or_none(gen: Generator[Any, None, None]) -> Any:
    try:
        return next(gen)
    except StopIteration:
        return None


# ── Path resolution ─────────────────────────────────────────────────────────

# Canonical Linux layout for the voice microservice:
#
#     /models/faster-whisper-large-v3/
#     /models/parakeet-tdt-0.6b-v3/
#     /models/kokoro-v0.19/
#     /models/xtts-v2/
#     /models/piper/{voice-id}.onnx (+ .json)
#
# The K8s deployment mounts the model PVC at /model-root and we
# symlink /models -> /model-root inside the container.

DEFAULT_MODEL_ROOT = "/models"


def resolve_model_dir(env_var: str, *fallbacks: str) -> str:
    """Return the first existing directory from env -> fallbacks -> "".
    
    Empty string means "no local copy; download from HuggingFace Hub on load".
    """
    import os
    candidate = os.environ.get(env_var, "").strip()
    if candidate and os.path.isdir(candidate):
        return candidate.rstrip("/\\")
        
    for fb in fallbacks:
        if not fb:
            continue
        if os.path.isdir(fb):
            return fb.rstrip("/\\")
    return ""


def resolve_hf_snapshot(path: str) -> str:
    """Resolve a HF cache directory to its concrete snapshot subdir.

    HF cache layout:
        models--org--name/
            blobs/
            refs/
            snapshots/<hash>/
                model.bin
                ...

    If the input is already a direct model directory, return as-is.
    """
    if not path:
        return ""
    if os.path.isfile(os.path.join(path, "model.bin")):
        return path
    snapshots = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots):
        versions = sorted(os.listdir(snapshots))
        if versions:
            resolved = os.path.join(snapshots, versions[-1])
            if os.path.isfile(os.path.join(resolved, "model.bin")):
                return resolved
    return path
