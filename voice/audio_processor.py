"""
Audio processing utilities for the voice conversation pipeline.

Handles:
    • PCM ↔ float conversion
    • Sample-rate conversion (48 kHz browser → 16 kHz Whisper)
    • Audio-level measurement (energy-based VAD backup)
    • Chunk assembly and normalisation
"""

from __future__ import annotations

import io
import struct
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE = 16_000      # Whisper expects 16 kHz mono
TTS_SAMPLE_RATE     = 24_000      # Kokoro outputs 24 kHz mono
BROWSER_SAMPLE_RATE = 48_000      # Typical browser AudioContext rate
INT16_MAX           = 32_767
INT16_MIN           = -32_768


# ── Conversion helpers ───────────────────────────────────────────────────────

def int16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit signed PCM bytes → float32 numpy array in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / INT16_MAX


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array [-1, 1] → raw 16-bit signed PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * INT16_MAX).astype(np.int16)
    return int16.tobytes()


def float32_bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
    """Convert raw little-endian float32 bytes → numpy float32 array."""
    return np.frombuffer(raw_bytes, dtype="<f4")


# ── Sample-rate conversion ───────────────────────────────────────────────────

def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple linear-interpolation resampler.

    Good enough for speech audio. For production, replace with
    ``librosa.resample`` if installed.
    """
    if from_rate == to_rate:
        return audio

    ratio = to_rate / from_rate
    new_length = int(len(audio) * ratio)
    if new_length == 0:
        return np.array([], dtype=np.float32)

    indices = np.arange(new_length) / ratio
    int_indices = indices.astype(int)
    frac = indices - int_indices

    # Clamp to valid range
    int_indices = np.clip(int_indices, 0, len(audio) - 1)
    next_indices = np.clip(int_indices + 1, 0, len(audio) - 1)

    resampled = audio[int_indices] * (1.0 - frac) + audio[next_indices] * frac
    return resampled.astype(np.float32)


def browser_audio_to_whisper(pcm_bytes: bytes, browser_rate: int = BROWSER_SAMPLE_RATE) -> np.ndarray:
    """Full pipeline: browser Int16 PCM → 16 kHz float32 for Whisper."""
    audio = int16_bytes_to_float32(pcm_bytes)
    if browser_rate != WHISPER_SAMPLE_RATE:
        audio = resample(audio, browser_rate, WHISPER_SAMPLE_RATE)
    return audio


# ── Audio-level measurement ──────────────────────────────────────────────────

def compute_rms(audio: np.ndarray) -> float:
    """Root Mean Square energy level of an audio chunk."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """RMS energy in decibels. Returns -∞ for silence."""
    rms = compute_rms(audio)
    if rms < 1e-10:
        return -100.0
    return float(20.0 * np.log10(rms / ref))


def is_speech(audio: np.ndarray, threshold_db: float = -35.0) -> bool:
    """Simple energy-based speech detection (server-side VAD backup)."""
    return compute_db(audio) > threshold_db


# ── Audio buffer ─────────────────────────────────────────────────────────────

class AudioBuffer:
    """Accumulates audio chunks and detects end-of-utterance.

    The buffer collects incoming PCM chunks until end-of-speech is detected
    (sustained silence following validated speech) or a manual flush is
    triggered. It then returns the complete utterance as a single float32
    numpy array at Whisper's expected sample rate.

    ── Noise-floor design (read before tuning) ──────────────────────────────
    Speech is gated on an adaptive floor: ``threshold = noise_floor + margin``.
    The single most important invariant is that **only sub-threshold audio may
    teach the floor**.

    A previous revision adapted the floor on every chunk until speech was
    *confirmed* (450 ms in), which meant each utterance's onset — and every
    chunk of assistant TTS echoing back through the phone speaker — dragged
    the estimate upward. Because speech is always louder than silence, this
    made the floor a one-way ratchet: production logs showed it climbing
    -52 dB → -21 dB over a single session until it crossed the threshold
    ceiling, at which point background noise itself scored as speech, silence
    never accumulated, and end-of-utterance stopped firing entirely (the
    session hung in "listening" forever).

    The fix here is twofold:
      1. Chunks at or above the speech threshold never touch the floor.
      2. The floor tracks asymmetrically — fast downward (a genuinely quieter
         room should be believed immediately) and slowly upward (steady noise
         creeping in), with time constants expressed in seconds so behaviour
         is independent of the caller's chunk size.

    The clamps are derived from measured phone-mic levels rather than picked
    by feel — see the constants below.

    Failure direction is a deliberate choice. In a room noisy enough to defeat
    an energy gate, these bounds bias toward *false positives* (background
    briefly scores as speech) rather than false negatives. A false positive
    costs one wasted STT call whose empty/hallucinated result is already
    filtered downstream in ``ws_handler``; a false negative silently wedges
    the session forever, which is the bug this class exists to prevent. An
    energy gate genuinely cannot separate speech from loud broadband noise —
    that is the boundary where an ML VAD (e.g. Silero) would be required.

    (``ws_handler`` is additionally responsible for not feeding assistant
    playback echo into this buffer at all — see its SPEAKING branch.)
    """

    # ── Level budget (dB RMS, measured on real phone-mic traffic) ────────
    #   quiet room      -70 .. -50      loud speech   ~-16
    #   noisy room      -50 .. -45      normal speech -20 .. -30
    #   quiet speech    -32 .. -35
    # Every bound below follows from that budget, and the invariant that the
    # gate must never demand speech louder than the quietest voice we intend
    # to hear (-35 dB), with headroom.

    # Bounds for the learned background level. The ceiling matters most: it
    # caps how far the gate can ever rise, which is what makes a wedged
    # session impossible.
    _FLOOR_MIN_DB = -70.0
    _FLOOR_MAX_DB = -45.0
    # How far above the floor a chunk must sit to count as speech.
    _SPEECH_MARGIN_DB = 8.0
    # Hard bounds on the resulting gate. The ceiling (_FLOOR_MAX + margin)
    # stays 2 dB below the quietest speech we expect; the base keeps a
    # digitally-silent stream from gating wide open.
    _THRESHOLD_MIN_DB = -55.0
    _THRESHOLD_MAX_DB = -37.0
    # Floor tracker time constants (seconds).
    _TAU_DOWN_S = 0.5
    _TAU_UP_S = 10.0
    # Audio retained while waiting for speech, so the first phoneme of an
    # utterance is never clipped off.
    _PREROLL_MS = 300

    def __init__(
        self,
        silence_duration_ms: int = 1000,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        max_duration_s: float = 30.0,
        min_speech_duration_ms: int = 450,
    ):
        self._chunks: list[np.ndarray] = []
        self._silence_samples = int(silence_duration_ms / 1000.0 * sample_rate)
        self._sample_rate = sample_rate
        self._max_samples = int(max_duration_s * sample_rate)
        self._min_speech_samples = int(min_speech_duration_ms / 1000.0 * sample_rate)
        self._preroll_samples = int(self._PREROLL_MS / 1000.0 * sample_rate)
        self._silent_samples_count = 0
        self._speech_samples_count = 0
        self._total_samples = 0
        self._speech_detected = False

        # Learned background level; persists across utterances (reset() keeps
        # it) so the gate stays converged for the whole session.
        self._noise_floor_db: Optional[float] = None

    # ── VAD state (exposed for logging and tests) ────────────────────────
    @property
    def noise_floor_db(self) -> Optional[float]:
        """Current learned background level, or ``None`` before the first chunk."""
        return self._noise_floor_db

    @property
    def speech_threshold_db(self) -> float:
        """dB level a chunk must exceed to count as speech."""
        if self._noise_floor_db is None:
            return self._THRESHOLD_MIN_DB
        return float(np.clip(
            self._noise_floor_db + self._SPEECH_MARGIN_DB,
            self._THRESHOLD_MIN_DB,
            self._THRESHOLD_MAX_DB,
        ))

    def add_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Add a chunk. Returns assembled audio if end-of-utterance detected.

        Returns ``None`` while speech is ongoing or no speech has started;
        returns the complete utterance float32 array once end-of-speech is
        detected or max duration is reached.
        """
        if len(audio) == 0:
            return None

        self._chunks.append(audio)
        self._total_samples += len(audio)

        chunk_s = len(audio) / float(self._sample_rate)
        current_db = compute_db(audio)

        # Seed the floor from the first chunk. If that chunk happens to be
        # speech the estimate starts too high, but the fast downward tracker
        # pulls it to the true floor within ~1 s of quiet.
        if self._noise_floor_db is None:
            self._noise_floor_db = float(np.clip(
                current_db, self._FLOOR_MIN_DB, self._FLOOR_MAX_DB,
            ))

        threshold = self.speech_threshold_db
        is_current_speech = current_db > threshold

        # ── Track the background level ───────────────────────────────────
        # Speech and echo are excluded entirely; see the class docstring.
        if not is_current_speech:
            tau = self._TAU_DOWN_S if current_db < self._noise_floor_db else self._TAU_UP_S
            alpha = float(np.exp(-chunk_s / tau))
            self._noise_floor_db = alpha * self._noise_floor_db + (1.0 - alpha) * current_db
            self._noise_floor_db = float(np.clip(
                self._noise_floor_db, self._FLOOR_MIN_DB, self._FLOOR_MAX_DB,
            ))

        # ── Speech / silence accounting ──────────────────────────────────
        if is_current_speech:
            self._silent_samples_count = 0
            self._speech_samples_count += len(audio)
            if not self._speech_detected and self._speech_samples_count >= self._min_speech_samples:
                self._speech_detected = True
                logger.info(
                    "Speech detected in buffer (accumulated %dms, noise floor: %.1fdB, threshold: %.1fdB)",
                    int(self._speech_samples_count / self._sample_rate * 1000),
                    self._noise_floor_db,
                    threshold,
                )
        elif self._speech_detected:
            self._silent_samples_count += len(audio)
        else:
            # Decay unconfirmed speech credit so isolated transients (a door,
            # a keyboard click) cannot accumulate into a phantom utterance.
            self._speech_samples_count = max(0, self._speech_samples_count - len(audio))
            # Nothing worth keeping yet: hold only a short pre-roll so idle
            # listening cannot grow the buffer or ship 30 s of silence to STT.
            self._trim_to_preroll()

        # End-of-utterance: sustained silence after validated speech.
        if self._speech_detected and self._silent_samples_count >= self._silence_samples:
            logger.info(
                "End-of-utterance detected (silence duration: %dms), flushing",
                int(self._silent_samples_count / self._sample_rate * 1000),
            )
            return self.flush()

        # Safety valve: a caller who never stops talking still gets processed.
        if self._total_samples >= self._max_samples:
            if self._speech_detected:
                logger.warning(
                    "Audio buffer hit max duration (%ds) with speech, flushing",
                    self._max_samples // self._sample_rate,
                )
                return self.flush()
            logger.info(
                "Audio buffer hit max duration (%ds) with no valid speech, silently resetting",
                self._max_samples // self._sample_rate,
            )
            self.reset()
            return None

        return None

    def _trim_to_preroll(self) -> None:
        """Drop buffered audio older than ``_PREROLL_MS`` while idle."""
        while self._total_samples > self._preroll_samples and len(self._chunks) > 1:
            self._total_samples -= len(self._chunks.pop(0))

    def flush(self) -> Optional[np.ndarray]:
        """Force-assemble all buffered chunks and reset."""
        if not self._chunks:
            return None
        assembled = np.concatenate(self._chunks)
        self.reset()
        return assembled

    def reset(self) -> None:
        """Clear the buffer. The learned noise floor is intentionally kept."""
        self._chunks.clear()
        self._silent_samples_count = 0
        self._speech_samples_count = 0
        self._total_samples = 0
        self._speech_detected = False

    @property
    def duration_s(self) -> float:
        """Current buffered audio duration in seconds."""
        return self._total_samples / self._sample_rate

    @property
    def has_audio(self) -> bool:
        return self._total_samples > 0
