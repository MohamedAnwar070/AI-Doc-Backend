# ============================================================
# Voice API Microservice — Linux, CPU-only, 3-model design
#
# Engines (exactly the three from VOICE_PLAN.md):
#   STT : Systran/faster-whisper-large-v3   (~3.0 GB)
#   TTS : Piper ONNX voices                 (~60 MB/voice)
#   VAD : snakers4/silero-vad (ONNX)        (~2 MB)
#
# CPU is the supported production path. PyTorch is pinned to the CPU
# index, no CUDA libs are pulled in, and Piper runs natively on CPU.
#
# Build:
#   docker build -f Dockerfile.voice \
#       --build-arg CPU_ONLY=true \
#       -t ai-doc-voice-model:v2.0 .
#
# Run (dev):
#   docker run --rm -p 8003:8003 \
#     -v /opt/models/voice:/models:ro \
#     ai-doc-voice-model:v2.0
# ============================================================

# ── CPU-only build flag ────────────────────────────────────────
# Set to false in a future GPU build to allow CUDA-enabled PyTorch.
ARG CPU_ONLY=true


# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

ARG CPU_ONLY=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# Build tools for compiling native audio/model dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        cmake \
        libsndfile1-dev \
        libffi-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch — CPU index by default, full CUDA index when CPU_ONLY=false.
RUN if [ "$CPU_ONLY" = "true" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch; \
    fi

# Install all voice dependencies from the dedicated requirements file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

ARG CPU_ONLY=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VOICE_CPU_ONLY=${CPU_ONLY}

WORKDIR /app

# Minimal runtime libs:
#   libgomp1     — OpenMP (required by torch / faster-whisper)
#   libsndfile1  — soundfile
#   libffi8      — cffi
#   ffmpeg       — used by /v1/audio/speech to encode mp3/opus/aac/flac
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
        libsndfile1 \
        libffi8 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and binaries from the builder.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code + voice engine package.
COPY --chown=1001:1001 voice_api.py ./
COPY --chown=1001:1001 voice ./voice
COPY --chown=1001:1001 config.py ./

# Model directory + non-root user.
# In Kubernetes, mount the model PVC at /models (see k8s/voice/deployment.yaml).
# In local dev, bind-mount /opt/models/voice to /models.
RUN mkdir -p /models && \
    useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app /models

USER 1001

# Default engine configuration. Override at runtime via env.
#
# All *DEVICE knobs are pinned to "cpu" because CPU is the supported
# production path. Override in a future GPU build with CPU_ONLY=false.
ENV VOICE_MODEL_ROOT=/models \
    WHISPER_MODEL_PATH=/models/faster-whisper-large-v3 \
    WHISPER_MODEL_SIZE=large-v3 \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    PIPER_MODEL_DIR=/models/piper \
    PIPER_DEFAULT_LANGUAGE=en \
    PIPER_DEFAULT_VOICE_EN=en_US-amy-low \
    PIPER_DEFAULT_VOICE_TA=ta_IN-rasa_female-medium \
    SILERO_VAD_ENABLED=true \
    SILERO_VAD_ONNX_PATH=/models/silero-vad/silero_vad.onnx \
    PORT=8003 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    CT2_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8003

# Health check — /health returns 200 even when no engines are loaded so a
# half-broken image still fails liveness in a useful way.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8003}/health || exit 1

CMD ["sh", "-c", "uvicorn voice_api:app --host 0.0.0.0 --port ${PORT:-8003} --workers 1"]
