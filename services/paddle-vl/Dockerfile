# ============================================================
# PDF OCR Backend — PaddleOCR-VL-1.5 (CPU) Dockerfile
# FastAPI + official PaddleOCR-VL document parser
# ============================================================
# CPU note:
#   PaddleOCR-VL works on CPU, but it is slow for full PDF workloads.
#   Use low concurrency and a high request timeout.
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM ubuntu:22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Python 3.10 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        gcc g++ \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the Paddle-VL microservice requirements first for layer caching.
COPY requirements-paddle-vl.txt ./requirements-paddle-vl.txt

# Step 1: Official PaddleOCR CPU runtime. Keep this aligned with
# ai-training/ocr-data/bootstrap_ground_truth.py, which uses PaddleOCRVL.
RUN pip install --default-timeout=300 "paddlepaddle==3.3.0" \
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Step 2: Other Python dependencies
RUN pip install --default-timeout=300 -r requirements-paddle-vl.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM ubuntu:22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
ENV OCR_ENGINE=paddle_vl
ENV GPU_ENABLED=false
ENV PADDLE_OCR_DEVICE=cpu
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

WORKDIR /app

# Minimal runtime OS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
        libsm6 libxext6 libxrender1 \
        libfreetype6 libharfbuzz0b \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL installed Python packages from builder (single location, no path confusion)
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Create data dirs and a non-root user first
RUN useradd -m -u 1001 appuser && \
    mkdir -p uploads outputs .cache/huggingface && \
    chown -R appuser:appuser /app

# Copy only the files needed by the standalone Paddle-VL API.
COPY --chown=appuser:appuser paddle_vl_api.py ./
COPY --chown=appuser:appuser ocr ./ocr

USER 1001

# PaddleOCRVL downloads/loads model files on first start. Mount PADDLE_VL_MODEL_DIR
# or the PaddleOCR cache volume in Kubernetes for persistent model storage.

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

EXPOSE 8001

CMD ["sh", "-c", "uvicorn paddle_vl_api:app --host 0.0.0.0 --port ${PORT:-8001}"]
