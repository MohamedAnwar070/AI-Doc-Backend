"""
Faster-Whisper STT engine.

Multilingual, including Tamil. Acts as the authoritative fallback for
every language and as the primary engine when Parakeet is disabled or
when the detected language is non-English.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import STTEngine, STTResult, resolve_hf_snapshot

logger = logging.getLogger(__name__)


# Whisper auto-detect override for bilingual en+ta deployments.
# Standard Whisper often mistakes Tamil/Tanglish accents for Spanish,
# Italian, Portuguese, etc. We force Tamil when the auto-detector returns
# any non-bilingual code.
_BILINGUAL_FORCE = {"en", "ta"}

# A short Tamil seed prompt. Passing it as ``initial_prompt`` biases the decoder
# toward Tamil script (rather than transliteration) when we transcribe in Tamil.
_TA_INITIAL_PROMPT = "இது ஒரு தமிழ் உரையாடல்."

# Decode parameters shared by every transcription pass. These are the guards
# against the classic Whisper failure modes on short/noisy phone-mic audio:
#   • temperature ladder      — retries greedy→sampled when a pass looks bad
#   • compression_ratio_thr.  — rejects passes whose text is suspiciously
#                               repetitive ("...தொடர்புக்கும் தொடர்புக்கும்...")
#   • log_prob_threshold      — rejects low-confidence hallucinated passes
#   • no_speech_threshold     — drops segments that are really silence
#   • repetition_penalty /    — actively discourage the decoder from looping
#     no_repeat_ngram_size      on the same n-gram
_ANTI_HALLUCINATION_PARAMS = dict(
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
)


class WhisperEngine(STTEngine):
    """faster-whisper (CTranslate2) wrapper.

    CPU is the supported production path — defaults to int8 quantisation
    which gives near-fp16 WER on CPU. Set device="cuda" + compute_type="float16"
    in a future GPU build to opt into GPU acceleration.
    """

    name = "whisper"
    kind = "stt"

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        compute_type: str = "int8",
        default_language: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.default_language = default_language
        self._model: Any = None

    # ── Capability ─────────────────────────────────────────────────────────
    def supports(self, language: Optional[str]) -> bool:
        # Whisper covers 99 languages.
        return True

    # ── Load ───────────────────────────────────────────────────────────────
    async def _load_impl(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        resolved = resolve_hf_snapshot(self.model_path) if self.model_path else ""

        device = self.device
        compute_type = self.compute_type

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "int8" if device == "cpu" else "float16"

        kwargs: Dict[str, Any] = dict(
            device=device,
            compute_type=compute_type,
            cpu_threads=min(os.cpu_count() or 4, 8),
        )

        if resolved:
            kwargs["model_size_or_path"] = resolved
        else:
            # Empty path -> fall back to default model size string.
            kwargs["model_size_or_path"] = "large-v3"

        logger.info(
            "Loading Faster-Whisper: path=%s device=%s compute_type=%s",
            kwargs["model_size_or_path"], device, compute_type,
        )

        self._model = WhisperModel(**kwargs)

        # Warm up so the first user request doesn't pay JIT cost.
        warmup = np.zeros(16000, dtype=np.float32)
        list(self._model.transcribe(warmup, language="en")[0])
        logger.info("Whisper model warmed up")

    # ── Transcribe ─────────────────────────────────────────────────────────
    def transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> STTResult:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded")

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        duration_s = len(audio) / float(sample_rate) if len(audio) else 0.0

        # RMS silence gate — same threshold as the legacy service.
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        db = 20.0 * np.log10(rms) if rms > 1e-10 else -100.0

        if duration_s < 0.25:
            return STTResult(
                text="",
                language=language or self.default_language or "en",
                engine=self.name,
                duration=duration_s,
            )

        if db < -42.0:
            logger.info("Whisper: skipping silent audio (%.1f dB)", db)
            return STTResult(
                text="",
                language=language or self.default_language or "en",
                engine=self.name,
                duration=duration_s,
            )

        t0 = time.time()

        lang_param = None if (language in (None, "", "auto")) else language
        use_vad = duration_s >= 1.0

        # Resolve the decode language up front. For a bilingual en+ta system we
        # never let Whisper wander into Spanish/Portuguese/Sanskrit (a frequent
        # mistake on Tamil/Tanglish accents — e.g. the logged 'sa' at p=0.41).
        # When the caller didn't force a language, restrict auto-detection to
        # {en, ta} *before* transcribing, instead of transcribing blind and
        # re-running a second full pass to correct it.
        forced = lang_param is not None
        decode_lang = lang_param if forced else self._detect_language_restricted(audio)
        initial_prompt = _TA_INITIAL_PROMPT if decode_lang == "ta" else None

        segs, info = self._run(audio, decode_lang, use_vad, initial_prompt)
        detected = decode_lang or info.language or (language or "en")

        # Retry once without VAD if it stripped everything.
        if not segs and use_vad and db > -34.0:
            logger.info("Whisper VAD removed all audio (%.1f dB); retrying without VAD", db)
            segs, info = self._run(audio, decode_lang, False, initial_prompt)
            detected = decode_lang or info.language or detected

        # Legacy safety net: if restricted detection was unavailable (older
        # faster-whisper without detect_language) and blind auto-detect returned
        # a non-bilingual language, force Tamil — the original behaviour — rather
        # than emit gibberish in the wrong script.
        if not forced and decode_lang is None and detected not in _BILINGUAL_FORCE:
            logger.info(
                "Whisper auto-detected '%s' (p=%.2f); forcing Tamil",
                detected, info.language_probability,
            )
            segs, info = self._run(audio, "ta", use_vad, _TA_INITIAL_PROMPT)
            if not segs and use_vad and db > -34.0:
                segs, info = self._run(audio, "ta", False, _TA_INITIAL_PROMPT)
            detected = "ta"

        if not segs:
            logger.info(
                "Whisper produced no segments: lang=%s duration=%.2fs level=%.1fdB prob=%.2f",
                detected, duration_s, db, info.language_probability,
            )
            return STTResult(
                text="",
                language=detected,
                language_probability=info.language_probability,
                duration=getattr(info, "duration", duration_s),
                engine=self.name,
                latency_ms=(time.time() - t0) * 1000,
            )

        segments: List[Dict[str, Any]] = []
        parts: List[str] = []
        for s in segs:
            t = (s.text or "").strip()
            if t:
                segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": t})
                parts.append(t)

        text = " ".join(parts).strip()
        return STTResult(
            text=text,
            language=info.language or detected,
            language_probability=info.language_probability,
            duration=getattr(info, "duration", duration_s),
            segments=segments,
            engine=self.name,
            latency_ms=(time.time() - t0) * 1000,
        )

    # ── Internal ───────────────────────────────────────────────────────────
    def _detect_language_restricted(self, audio: np.ndarray) -> Optional[str]:
        """Detect the language but restrict the answer to {en, ta}.

        Returns ``"en"`` or ``"ta"`` (whichever scores higher among the two),
        or ``None`` when the running faster-whisper build predates the
        ``detect_language`` API — in which case the caller falls back to the
        legacy blind-auto-detect-then-force-Tamil path.
        """
        detect = getattr(self._model, "detect_language", None)
        if detect is None:
            return None
        try:
            result = detect(audio)
        except Exception as exc:  # pragma: no cover — version/shape guard
            logger.debug("detect_language unavailable/failed: %s", exc)
            return None

        # faster-whisper returns (language, probability, all_language_probs).
        all_probs = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
        if not all_probs:
            return None

        probs: Dict[str, float] = {}
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
            "Restricted language detection → %s (p_en=%.2f, p_ta=%.2f)",
            chosen, p_en, p_ta,
        )
        return chosen

    def _run(
        self,
        audio: np.ndarray,
        lang: Optional[str],
        use_vad: bool,
        initial_prompt: Optional[str] = None,
    ):
        return self._model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            best_of=5,
            patience=1.0,
            vad_filter=use_vad,
            vad_parameters=dict(min_silence_duration_ms=700, speech_pad_ms=300),
            condition_on_previous_text=False,
            suppress_blank=True,
            initial_prompt=initial_prompt,
            **_ANTI_HALLUCINATION_PARAMS,
        )
