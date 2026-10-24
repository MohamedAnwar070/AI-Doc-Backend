"""
Silero VAD engine.

Replaces Whisper's internal VAD as the front gate for end-of-speech
detection. ~1 ms per 30 ms chunk on CPU, much more accurate than
energy-threshold VAD on quiet / reverberant audio.

Reference: https://github.com/snakers4/silero-vad
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np

from .base import VADEngine, VADSegment

logger = logging.getLogger(__name__)


class SileroVADEngine(VADEngine):
    """Silero VAD using a locally managed ONNX model."""

    name = "silero_vad"
    kind = "vad"

    def __init__(
        self,
        onnx_path: str = "/models/silero-vad/silero_vad.onnx",
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
        speech_pad_ms: int = 300,
    ) -> None:
        super().__init__()
        self.onnx_path = onnx_path
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self._model: Optional[Any] = None

    # ── Capability ─────────────────────────────────────────────────────────
    def supports(self, language: object) -> bool:
        # VAD is language-agnostic.
        return True

    # ── Load ───────────────────────────────────────────────────────────────
    async def _load_impl(self) -> None:
        try:
            # The public loader only accepts ``onnx=True`` and loads the
            # wheel's bundled model. This upstream wrapper is required for
            # the explicitly configured model used by offline deployments.
            from silero_vad.utils_vad import OnnxWrapper  # type: ignore
            import onnxruntime  # noqa: F401  # ensure dep present
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad not installed. pip install silero-vad onnxruntime"
            ) from exc

        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(
                f"Silero VAD ONNX weights not found at {self.onnx_path}. "
                "Download from https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            )

        self._model = OnnxWrapper(self.onnx_path, force_onnx_cpu=True)
        logger.info("Silero VAD loaded from %s", self.onnx_path)

    # ── Inference ──────────────────────────────────────────────────────────
    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """Return True if the 30 ms or 60 ms mono chunk contains speech."""
        if self._model is None:
            return True  # fail-open: don't block audio if VAD is broken
        import torch

        chunk_size = 512 if sample_rate == 16000 else 256
        if len(audio_chunk) != chunk_size:
            # Pad/trim so we always feed the expected size.
            if len(audio_chunk) < chunk_size:
                audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:chunk_size]
        tensor = torch.from_numpy(audio_chunk.astype(np.float32, copy=False))
        prob = self._model(tensor, sample_rate).item()
        return float(prob) >= self.threshold

    def segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_silence_ms: int = 700,
        speech_pad_ms: int = 300,
    ) -> List[VADSegment]:
        """Split ``audio`` into speech regions using Silero VAD."""
        if self._model is None:
            return [VADSegment(0.0, len(audio) / float(sample_rate))]

        if audio.size == 0:
            return []

        import torch
        from silero_vad import get_speech_timestamps  # type: ignore

        timestamps = get_speech_timestamps(
            torch.from_numpy(audio.astype(np.float32, copy=False)),
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            sampling_rate=sample_rate,
            return_seconds=True,
        )
        return [
            VADSegment(float(item["start"]), float(item["end"]))
            for item in timestamps
        ]
