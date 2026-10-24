# Realtime Voice Assistant — Setup & Deployment Guide

This package adds a modern, low-latency voice conversation layer to the document OCR + RAG platform. Users talk directly to the system in **English, Tamil, or Tanglish (mixed)** and hear the reply in the same language. The pipeline runs entirely on CPU with **3 open-source HuggingFace models**.

**v2.0 (2026-06)** — Single STT + single TTS + VAD. See [`VOICE_PLAN.md`](VOICE_PLAN.md) for the full design rationale.

---

## What's in the Box

| Role | Model | Size | Source |
|------|-------|------|--------|
| **STT** | `Systran/faster-whisper-large-v3` | 3.0 GB | HF Hub (open-source) |
| **TTS** | `coqui/XTTS-v2` | 2.0 GB | HF Hub (open-source) |
| **VAD** | `snakers4/silero-vad` (ONNX) | 2 MB | GitHub (open-source) |

That's it. **3 models, ~5 GB on disk.** Whisper handles Tamil + English + Tanglish code-switching in one model; XTTS does the same on the synthesis side, with built-in voice cloning.

---

## Layout

```
/opt/models/voice/
├── faster-whisper-large-v3/      ← Whisper weights
├── xtts-v2/                      ← XTTS weights + speakers/ subdir
└── silero-vad/
    └── silero_vad.onnx           ← Silero VAD ONNX
```

---

## One-Time Setup

```bash
cd /opt/ai-doc-retrieval-backend

# 1. Download all 3 models (idempotent, ~5 GB)
.venv/bin/python download_voice_models.py

# 2. Start the voice microservice
.venv/bin/python voice_api.py
# → listening on http://0.0.0.0:8003

# 3. (other shell) start the main backend
.venv/bin/python main.py
# → log shows: NEED_VOICE is enabled. Pre-loading voice models...
```

That's the entire setup. No env juggling, no separate model servers per engine.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Main Backend (port 8000) — ai-doc-retrieval-backend             │
│    ├─ voice/ws_handler.py          — /ws/voice (WebSocket)       │
│    ├─ voice/stt_service.py         — POSTs PCM to microservice   │
│    ├─ voice/tts_service.py         — streams TTS from microservice│
│    ├─ voice/conversation_engine.py — RAG + LLM (unchanged)       │
│    └─ voice/audio_processor.py     — PCM math (unchanged)        │
└─────────────────────────┬────────────────────────────────────────┘
                          │  HTTP (VOICE_MODEL_URL=http://localhost:8003)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  Voice API Microservice (port 8003) — voice_api.py               │
│                                                                  │
│  /health                — engine status                          │
│  /v1/engines            — loaded engine map                      │
│  /v1/voices             — voice catalog                          │
│  /v1/audio/transcriptions  — OpenAI-compatible STT               │
│  /v1/audio/speech          — OpenAI-compatible TTS               │
│  /transcribe            — legacy raw-PCM STT                     │
│  /tts                   — legacy NDJSON TTS                      │
│  /v1/voices/clone       — register voice clone                   │
│                                                                  │
│  Loaded engines:                                                 │
│    STT  → Whisper large-v3   (handles ta / en / Tanglish)        │
│    TTS  → XTTS-v2            (handles ta / en / Tanglish + clone)│
│    VAD  → Silero v4          (end-of-speech detection)           │
└──────────────────────────────────────────────────────────────────┘
                          ▲
                          │  bind-mount: /opt/models/voice
                          │  K8s: PVC at /model-root
```

---

## Configuration (.env)

```env
NEED_VOICE=true
VOICE_MODEL_URL=http://localhost:8003
VOICE_API_KEY=

# STT
WHISPER_MODEL_PATH=/opt/models/voice/faster-whisper-large-v3
WHISPER_MODEL_SIZE=large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# TTS
XTTS_ENABLED=true
XTTS_MODEL_PATH=/opt/models/voice/xtts-v2
XTTS_SPEAKER_DIR=/opt/models/voice/xtts-v2/speakers
XTTS_DEFAULT_LANGUAGE=en          # switch to "ta" for Tamil-dominant replies

# VAD
SILERO_VAD_ENABLED=true
SILERO_VAD_ONNX_PATH=/opt/models/voice/silero-vad/silero_vad.onnx

# CPU threading (match your pod CPU limit)
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
CT2_NUM_THREADS=8

# Root for all model weights
VOICE_MODEL_ROOT=/opt/models/voice
```

---

## API Examples

### STT — OpenAI-compatible

```bash
curl -X POST http://localhost:8003/v1/audio/transcriptions \
     -F file=@sample.wav \
     -F language=en \
     -F response_format=verbose_json
```

Returns:

```json
{
  "text": "Hello, how are you?",
  "language": "en",
  "engine": "whisper",
  "latency_ms": 412.3
}
```

For Tamil: pass `language=ta` or omit for auto-detect. Whisper handles the Tanglish middle ground gracefully — it returns the script it heard.

### TTS — OpenAI-compatible

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
     -H "content-type: application/json" \
     -d '{"input":"Hello, how can I help you?","voice":"default","response_format":"mp3"}' \
     --output test.mp3
```

For Tamil: set `XTTS_DEFAULT_LANGUAGE=ta` (in env) or pass `language=ta` in the request body.

### Voice Cloning (XTTS)

```bash
# Register a 6–30 second reference clip
curl -X POST http://localhost:8003/v1/voices/clone \
     -F voice=default \
     -F file=@my_voice.wav

# Synthesise using the cloned voice
curl -X POST http://localhost:8003/v1/audio/speech \
     -H "content-type: application/json" \
     -d '{"input":"Hello, this is Tamil: வணக்கம்","voice":"clone:custom_speaker"}' \
     --output cloned.wav
```

---

## Language Strategy

| Input | STT does | TTS does |
|-------|----------|----------|
| English | Whisper (`language=en` or auto) | XTTS with `language=en` |
| Tamil | Whisper (`language=ta`) | XTTS with `language=ta` |
| Tanglish | Whisper auto-detects, returns whatever script it heard | XTTS renders mixed text — set `XTTS_DEFAULT_LANGUAGE` to the dominant script |

XTTS-v2 is trained on a 16-language multilingual corpus that includes Tamil. It accepts a `language` parameter per call and handles intra-utterance code-switching gracefully.

---

## Deployment (Kubernetes)

```bash
# 1. PVC for the 3 models (10 GB)
kubectl apply -f k8s/voice/pvc.yaml

# 2. Build the voice image (CPU-only, no CUDA libs)
docker build -f Dockerfile.voice \
    --build-arg CPU_ONLY=true \
    -t ai-doc-voice-model:v2.0 .
docker push ai-doc-voice-model:v2.0

# 3. Roll out
kubectl apply -f k8s/voice/deployment.yaml
kubectl apply -f k8s/voice/service.yaml
kubectl apply -f k8s/voice/ingress.yaml

# 4. Verify
curl http://localhost:8003/health | jq .
```

The K8s deployment requests 6 GiB RAM / 4 CPU, limits 14 GiB / 8 CPU — same tier as `paddle-vl`. PyTorch is pinned to the CPU index inside the image, so no `nvidia.com/gpu` resource is requested.

---

## CPU-First Architecture (Production Default)

Every model in the v2 stack is open-source and runs natively on CPU:

| Engine | Backend | Open-source | CPU default |
|--------|---------|-------------|-------------|
| Whisper large-v3 | CTranslate2 int8 | ✅ | ✅ |
| XTTS-v2 | PyTorch CPU | ✅ | ✅ (~10× realtime) |
| Silero VAD v4 | onnxruntime CPU | ✅ | ✅ |

Enforcement:
- `Dockerfile.voice` installs PyTorch from `https://download.pytorch.org/whl/cpu` when `CPU_ONLY=true` (default).
- All `WHISPER_DEVICE` env defaults are `cpu` + `compute_type=int8`.
- XTTS runs on CPU without any device flag (Coqui TTS auto-selects).

GPU support is deferred to a future build. To enable it: `docker build --build-arg CPU_ONLY=false ...` and override the device env vars. No code changes needed.

---

## Key Implementation Files

### Backend (main process — port 8000)
- [`voice/ws_handler.py`](voice/ws_handler.py) — WebSocket endpoint `/ws/voice`
- [`voice/stt_service.py`](voice/stt_service.py) — routes STT to remote microservice
- [`voice/tts_service.py`](voice/tts_service.py) — routes TTS to remote microservice
- [`voice/conversation_engine.py`](voice/conversation_engine.py) — two-pass LLM + RAG
- [`voice/audio_processor.py`](voice/audio_processor.py) — PCM math
- [`voice/session_manager.py`](voice/session_manager.py) — session state + barge-in

### Voice Microservice (port 8003)
- [`voice_api.py`](voice_api.py) — FastAPI app, loads Whisper + XTTS + Silero VAD
- [`voice/engines/base.py`](voice/engines/base.py) — abstract engine contracts
- [`voice/engines/router.py`](voice/engines/router.py) — engine picker (degenerates to single-engine since v2 has one per role)
- [`voice/engines/stt_whisper.py`](voice/engines/stt_whisper.py) — Whisper wrapper
- [`voice/engines/tts_xtts.py`](voice/engines/tts_xtts.py) — XTTS-v2 wrapper
- [`voice/engines/vad_silero.py`](voice/engines/vad_silero.py) — Silero VAD

### Deployment
- [`Dockerfile.voice`](Dockerfile.voice) — CPU-only Linux image
- [`k8s/voice/deployment.yaml`](k8s/voice/deployment.yaml) — K8s Deployment
- [`k8s/voice/pvc.yaml`](k8s/voice/pvc.yaml) — 10 GB PVC for the 3 models
- [`k8s/voice/service.yaml`](k8s/voice/service.yaml) — ClusterIP service
- [`k8s/voice/ingress.yaml`](k8s/voice/ingress.yaml) — TLS ingress
- [`download_voice_models.py`](download_voice_models.py) — model downloader
- [`VOICE_PLAN.md`](VOICE_PLAN.md) — the full design plan

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: faster_whisper` | `pip install faster-whisper` |
| `OSError: XTTS model not found` | `python download_voice_models.py /opt/models/voice` |
| `/health` shows `xtts: not loaded` | XTTS download incomplete — re-run downloader; check disk space |
| Whisper ignores Tamil | pass `language=ta` explicitly, or remove `language` for auto-detect |
| XTTS pronounces Tamil with English phonemes | set `XTTS_DEFAULT_LANGUAGE=ta` for Tamil-dominant replies |
| Out of memory in K8s pod | bump `resources.limits.memory` to 16Gi |
