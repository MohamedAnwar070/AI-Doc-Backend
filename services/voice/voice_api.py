"""
Standalone Voice Microservice — Linux-native, simplified 3-model design.

Architecture (mirrors paddle_vl_api.py):
    • FastAPI app on port 8003
    • Models mounted read-only at /models (Linux path) or
      /opt/models/voice (the user-configured root)
    • Engines loaded in parallel during the lifespan startup phase
    • Legacy /transcribe + /tts endpoints kept for backwards compat
    • OpenAI-compatible /v1/audio/transcriptions and /v1/audio/speech
      endpoints for 3rd-party integration
    • /v1/voices for the TTS voice catalog

Engines (exactly three — see VOICE_PLAN.md):
    STT  : Systran/faster-whisper-large-v3       (always)
    TTS  : Piper ONNX (rhasspy/piper-voices)     (always)
            English: en_US-amy-low
            Tamil:   ta_IN-vits-medium
    VAD  : snakers4/silero-vad (ONNX)            (always)

Endpoints
---------
GET  /health
GET  /v1/engines
GET  /v1/voices
POST /v1/audio/transcriptions           (OpenAI-compatible, multipart upload)
POST /v1/audio/speech                   (OpenAI-compatible, returns audio)
POST /transcribe                        (legacy: raw PCM bytes, returns JSON)
POST /tts                               (legacy: NDJSON stream of base64 PCM)
POST /v1/voices/clone                   (Piper doesn't support cloning — returns 501)

Configuration (env vars)
------------------------
WHISPER_MODEL_PATH          Default: /opt/models/voice/faster-whisper-large-v3
WHISPER_MODEL_SIZE          Default: large-v3
WHISPER_DEVICE              Default: cpu
WHISPER_COMPUTE_TYPE        Default: int8
PIPER_MODEL_DIR             Default: /opt/models/voice/piper
PIPER_DEFAULT_LANGUAGE      Default: en          (switch to "ta" for Tamil replies)
PIPER_DEFAULT_VOICE_EN      Default: en_US-amy-low
PIPER_DEFAULT_VOICE_TA      Default: ta_IN-vits-medium
SILERO_VAD_ONNX_PATH        Default: /opt/models/voice/silero-vad/silero_vad.onnx
VOICE_API_KEY               Optional bearer token for non-/health routes
PORT                        Default: 8003
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import platform
import re
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    Depends,
    Security,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voice_api")

# ── CONSTANTS ────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE = 16_000
INT16_MAX = 32_767

MODEL_ROOTS_BY_OS = {
    "Darwin": "/Users/gsmmgsm/gsm/ai-ml/ai-orginal/ai-models/voice-models",
    "Linux": "/opt/models/voice-models",
    "Windows": r"D:\models\voice-models",
}
DEFAULT_MODEL_ROOT = MODEL_ROOTS_BY_OS.get(platform.system(), "/models")
VOICE_MODEL_ROOT = os.environ.get("VOICE_MODEL_ROOT", DEFAULT_MODEL_ROOT)


# ── Optional API-key guard ──────────────────────────────────────────────────
API_KEY = os.environ.get("VOICE_API_KEY")
security = HTTPBearer(auto_error=False)

def verify_api_key(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY:
        return None

    header = request.headers.get("Authorization")
    if not header:
        raise HTTPException(status_code=403, detail="Invalid authentication credentials")

    if header.startswith("Bearer "):
        token = header.split(" ")[1]
        if token == API_KEY:
            return token

    elif header.startswith("Basic "):
        try:
            import base64
            decoded = base64.b64decode(header.split(" ")[1]).decode()
            if API_KEY in decoded.split(":"):
                return API_KEY
        except Exception:
            pass

    logger.warning("Unauthorized access attempt.")
    raise HTTPException(status_code=403, detail="Invalid authentication credentials")


# ── Audio helpers ────────────────────────────────────────────────────────────
def int16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / INT16_MAX


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio is None or len(audio) == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * INT16_MAX).astype(np.int16).tobytes()


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono as a 16-bit PCM WAV."""
    import io
    import struct

    if audio is None or len(audio) == 0:
        audio = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
    int16 = (np.clip(audio, -1.0, 1.0) * INT16_MAX).astype(np.int16)
    data_size = int16.nbytes
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(int16.tobytes())
    return buf.getvalue()


def write_upload_to_temp(upload: UploadFile) -> str:
    """Save an UploadFile to a temp file and return the path."""
    import tempfile

    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def load_audio_file(path: str, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Load any audio file (wav/mp3/flac/ogg) and resample to mono float32."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa  # type: ignore
        audio, sr = librosa.load(path, sr=None, mono=True)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    if sr != target_sr:
        duration = len(audio) / sr
        new_len = max(int(duration * target_sr), 1)
        xp = np.linspace(0.0, duration, num=len(audio), endpoint=True)
        xq = np.linspace(0.0, duration, num=new_len, endpoint=True)
        audio = np.interp(xq, xp, audio).astype(np.float32)

    return audio


# ── Engine bootstrap (only the 3 engines in VOICE_PLAN.md) ──────────────────
async def _load_engine(engine) -> None:
    """Wrap ``engine.load()`` so a single failure does not stop the others."""
    try:
        await engine.load()
    except Exception as exc:
        logger.error("Engine %s failed to load: %s", engine.name, exc, exc_info=True)


async def _build_registry() -> None:
    """Construct + load Whisper, Piper, and Silero VAD in parallel."""
    from voice.engines import router as engine_router
    from voice.engines.base import resolve_model_dir
    from voice.engines.stt_whisper import WhisperEngine
    from voice.engines.tts_piper import PiperTTSEngine
    from voice.engines.vad_silero import SileroVADEngine

    registry = engine_router.get_registry()
    load_tasks: List[asyncio.Task] = []

    # Resolve the model root once. All three models share the same parent.
    model_root = (
        os.environ.get("VOICE_MODEL_ROOT")
        or VOICE_MODEL_ROOT
    )

    # ── STT: Whisper (always loaded) ───────────────────────────────────────
    whisper_path = os.environ.get("WHISPER_MODEL_PATH") or os.path.join(model_root, "faster-whisper-large-v3")
    whisper_engine = WhisperEngine(
        model_path=whisper_path,
        device=os.environ.get("WHISPER_DEVICE", "cpu"),
        compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
        default_language=os.environ.get("WHISPER_DEFAULT_LANGUAGE", None),
    )
    registry.register_stt(whisper_engine)
    load_tasks.append(asyncio.create_task(_load_engine(whisper_engine)))

    # ── TTS: Piper ONNX (always loaded) ────────────────────────────────────
    piper_dir = os.environ.get("PIPER_MODEL_DIR") or os.path.join(model_root, "piper")
    piper_engine = PiperTTSEngine(
        model_dir=piper_dir,
        default_voice_en=os.environ.get("PIPER_DEFAULT_VOICE_EN", "en_US-amy-low"),
        default_voice_ta=os.environ.get("PIPER_DEFAULT_VOICE_TA", "ta_IN-vits-medium"),
        default_language=os.environ.get("PIPER_DEFAULT_LANGUAGE", "en"),
    )
    registry.register_tts(piper_engine)
    load_tasks.append(asyncio.create_task(_load_engine(piper_engine)))

    # ── VAD: Silero v4 (always loaded) ─────────────────────────────────────
    vad_path = os.environ.get("SILERO_VAD_ONNX_PATH") or os.path.join(model_root, "silero-vad", "silero_vad.onnx")

    vad_engine = SileroVADEngine(
        onnx_path=vad_path,
    )
    registry.register_stt(vad_engine)  # registered alongside STT for visibility
    load_tasks.append(asyncio.create_task(_load_engine(vad_engine)))

    if load_tasks:
        logger.info("Loading %d voice engine(s) in parallel\u2026", len(load_tasks))
        await asyncio.gather(*load_tasks, return_exceptions=True)
        loaded = [e.name for e in registry.all_engines if e.is_loaded]
        failed = [(e.name, e.load_error) for e in registry.all_engines if not e.is_loaded]
        logger.info("Voice engines loaded: %s. Failed: %s", loaded, failed or "none")


# ── Sentence splitting (for sentence-streaming TTS) ─────────────────────────
_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+|(?<=[.!?।])$|(?<=\n)\s*", re.UNICODE)


def _split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = _SENTENCE_RE.split(text.strip())
    sentences = [s.strip() for s in parts if s and s.strip()]
    if not sentences:
        return [text.strip()]
    merged: List[str] = []
    for s in sentences:
        if merged and len(s) < 15 and not s.endswith((".", "!", "?", "।")):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


# ── Request models ──────────────────────────────────────────────────────────
class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    language: Optional[str] = None
    response_format: Optional[str] = "pcm"


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech payload."""

    model: str = "voice-default"
    input: str = Field(..., description="Text to synthesise")
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0


# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    logger.info("=" * 60)
    logger.info("Voice API starting on port %s (CPU_ONLY=%s)",
                os.environ.get("VOICE_PORT", "8003"),
                os.environ.get("VOICE_CPU_ONLY", "true"))
    logger.info("Model root: %s", os.environ.get("VOICE_MODEL_ROOT", DEFAULT_MODEL_ROOT))
    logger.info("=" * 60)
    await _build_registry()
    yield
    logger.info("Voice API shutting down.")


app = FastAPI(
    title="Voice API Microservice",
    version="2.0.0",
    description="Linux-native, CPU-only voice pipeline: Whisper large-v3 + Piper + Silero VAD.",
    lifespan=lifespan,
)


# ── Health & catalog ────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> Dict[str, Any]:
    from voice.engines import router as engine_router
    registry = engine_router.get_registry()
    engines = registry.snapshot()
    return {
        "status": "ok",
        "loaded": [e["name"] for e in engines if e["loaded"]],
        "failed": [{"name": e["name"], "error": e["error"]} for e in engines if not e["loaded"]],
        "engines": engines,
        "model_root": os.environ.get("VOICE_MODEL_ROOT", DEFAULT_MODEL_ROOT),
        "cpu_only": os.environ.get("VOICE_CPU_ONLY", "true"),
    }


@app.get("/v1/engines")
async def v1_engines(request: Request, api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    from voice.engines import router as engine_router
    registry = engine_router.get_registry()
    return {"data": registry.snapshot()}


@app.get("/v1/voices")
async def v1_voices(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    from voice.engines import router as engine_router
    registry = engine_router.get_registry()
    return {"data": registry.list_voices()}


# ── STT helpers ─────────────────────────────────────────────────────────────
async def _transcribe_with_engine(
    audio: np.ndarray,
    language: Optional[str],
):
    from voice.engines import router as engine_router
    registry = engine_router.get_registry()

    # Only Whisper is loaded for STT; ignore preferred_engine name (kept for API compat).
    for e in registry.stt_engines:
        if e.name == "whisper":
            return await e.transcribe(audio, WHISPER_SAMPLE_RATE, language=language)

    raise HTTPException(status_code=503, detail="Whisper STT model not loaded")


# ── STT endpoints ───────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe_legacy(
    request: Request,
    language: Optional[str] = Query(None, description="Force language (ta|en) or 'auto'"),
    api_key: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Legacy raw-PCM endpoint used by voice/stt_service.py."""
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    audio = int16_bytes_to_float32(audio_bytes)
    if len(audio) == 0:
        return {"text": "", "language": language or "en", "segments": []}
    result = await _transcribe_with_engine(audio, language=language)
    return {
        "text": result.text,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "segments": result.segments,
        "engine": result.engine,
        "latency_ms": result.latency_ms,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_openai(
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/ogg/m4a)"),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    api_key: str = Depends(verify_api_key),
) -> Response:
    """OpenAI-compatible STT endpoint."""
    from voice.engines import router as engine_router
    registry = engine_router.get_registry()

    tmp_path = write_upload_to_temp(file)
    try:
        audio = load_audio_file(tmp_path, target_sr=WHISPER_SAMPLE_RATE)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if len(audio) == 0:
        if response_format == "text":
            return PlainTextResponse("")
        return JSONResponse({"text": "", "language": language or "en", "segments": []})

    result = await _transcribe_with_engine(audio, language=language)

    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "srt":
        return PlainTextResponse(_segments_to_srt(result.segments))
    if response_format == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": result.language,
            "duration": result.duration,
            "text": result.text,
            "segments": [
                {
                    "id": i,
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
                for i, s in enumerate(result.segments)
            ],
            "engine": result.engine,
            "latency_ms": result.latency_ms,
        })
    return JSONResponse({
        "text": result.text,
        "language": result.language,
        "engine": result.engine,
        "latency_ms": result.latency_ms,
    })


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_timestamp(seg["start"])
        end = _format_srt_timestamp(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    return "\n".join(lines)


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── TTS helpers ─────────────────────────────────────────────────────────────
async def _synthesise_streaming(
    text: str,
    voice: str = "default",
    speed: float = 1.0,
    language: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    from voice.engines import router as engine_router
    registry = engine_router.get_registry()

    # Only Piper is loaded for TTS.
    for e in registry.tts_engines:
        if e.name == "piper":
            engine = e
            break
    else:
        raise HTTPException(status_code=503, detail="Piper TTS model not loaded")

    sentences = _split_sentences(text) or [text]

    async for chunk in engine.synthesise(text, voice=voice, speed=speed, language=language):
        yield {
            "text": chunk.text,
            "sample_rate": chunk.sample_rate,
            "engine": chunk.engine,
            "audio_int16_b64": base64.b64encode(chunk.to_int16_bytes()).decode("ascii"),
        }


# ── TTS endpoints ───────────────────────────────────────────────────────────
@app.post("/tts")
async def tts_legacy(request: TtsRequest, api_key: str = Depends(verify_api_key)) -> StreamingResponse:
    """Legacy NDJSON stream used by voice/tts_service.py."""
    async def _stream() -> AsyncGenerator[str, None]:
        async for item in _synthesise_streaming(
            text=request.text,
            voice=request.voice or "default",
            speed=request.speed,
            language=request.language,
        ):
            payload = {
                "text": item["text"],
                "sample_rate": item["sample_rate"],
                "engine": item["engine"],
                "audio_b64": item["audio_int16_b64"],
            }
            yield json.dumps(payload) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/v1/audio/speech")
async def speech_openai(req: SpeechRequest, api_key: str = Depends(verify_api_key)) -> Response:
    """OpenAI-compatible TTS endpoint."""
    int16_chunks: List[np.ndarray] = []
    sample_rate: Optional[int] = None
    async for item in _synthesise_streaming(
        text=req.input,
        voice=req.voice,
        speed=req.speed,
    ):
        if sample_rate is None:
            sample_rate = item["sample_rate"]
        pcm = base64.b64decode(item["audio_int16_b64"])
        int16_chunks.append(np.frombuffer(pcm, dtype=np.int16))

    if not int16_chunks or sample_rate is None:
        raise HTTPException(status_code=503, detail="TTS produced no audio")

    combined = np.concatenate(int16_chunks)
    audio_float = combined.astype(np.float32) / INT16_MAX

    fmt = req.response_format.lower()
    if fmt == "wav":
        return Response(content=wav_bytes(audio_float, sample_rate), media_type="audio/wav")
    if fmt == "pcm":
        return Response(content=combined.tobytes(), media_type="audio/pcm")
    if fmt in ("mp3", "opus", "aac", "flac"):
        encoded = _try_ffmpeg_transcode(audio_float, sample_rate, fmt)
        if encoded is not None:
            media = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
            }[fmt]
            return Response(content=encoded, media_type=media)
        logger.warning("ffmpeg unavailable; returning wav instead of %s", fmt)
        return Response(content=wav_bytes(audio_float, sample_rate), media_type="audio/wav")

    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {req.response_format}")


def _try_ffmpeg_transcode(audio: np.ndarray, sample_rate: int, fmt: str) -> Optional[bytes]:
    """Use ffmpeg to encode audio into mp3/opus/aac/flac. Returns None if ffmpeg is missing."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_path = in_f.name
    out_path = in_path + f".{fmt}"
    try:
        with open(in_path, "wb") as f:
            f.write(wav_bytes(audio, sample_rate))
        codec_map = {
            "mp3": ["-codec:a", "libmp3lame", "-qscale:a", "2"],
            "opus": ["-codec:a", "libopus", "-b:a", "64k"],
            "aac": ["-codec:a", "aac", "-b:a", "128k"],
            "flac": ["-codec:a", "flac"],
        }
        args = ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path] + codec_map[fmt] + [out_path]
        rc = subprocess.run(args, check=False).returncode
        if rc != 0:
            return None
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Voice cloning (NOT SUPPORTED — Piper has no voice-clone API) ───────────
@app.post("/v1/voices/clone")
async def voice_clone(
    voice: str = Form(..., description="Unused — Piper doesn't support voice cloning."),
    file: UploadFile = File(..., description="Unused — Piper doesn't support voice cloning."),
    api_key: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Voice cloning is unavailable with Piper ONNX. Returns 501.

    To change the TTS voice, drop a Piper ``.onnx`` + ``.onnx.json`` pair
    into ``PIPER_MODEL_DIR`` and reference it via the ``voice`` field on
    /v1/audio/speech (e.g. ``"voice": "ta_IN-vits-medium"``).
    """
    raise HTTPException(
        status_code=501,
        detail="Voice cloning is not supported by the Piper TTS engine. "
               "Pre-trained voices are bundled; drop new .onnx files into PIPER_MODEL_DIR to add more.",
    )


# ── CLI entry point ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Local Voice API microservice (Whisper + Piper + Silero VAD)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VOICE_PORT", "8003")))
    args = parser.parse_args()

    uvicorn.run(
        "voice_api:app",
        host=args.host,
        port=args.port,
        workers=1,  # engines are in-process singletons
        log_level="info",
    )


if __name__ == "__main__":
    main()
