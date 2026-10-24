# ============================================================
# PDF OCR Backend - Local Embedding API Dockerfile
# FastAPI + SentenceTransformers (CPU-optimized)
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build tools for compiling native dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and torchvision CPU first to completely blockade CUDA downloads
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies compatible with the exported embedding model.
RUN pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 pydantic sentence-transformers==5.4.1 transformers==4.57.1 huggingface-hub==0.36.0

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 
    # LOCAL_EMBEDDING_MODEL_PATH=all-MiniLM-L6-v2-fixed/

WORKDIR /app

# Minimal runtime OS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application script
COPY embedding_api.py .

# Create model directory and a non-root user
# Note: When deploying with K8s, mount your PVC to /models, 
# ensuring that the folder /models/all-MiniLM-L6-v2 exists inside it.
RUN mkdir -p /models && \
    useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app /models

USER appuser

# Health check matching embedding_api.py health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8002}/health || exit 1

EXPOSE 8002

# Startup command
CMD ["sh", "-c", "uvicorn embedding_api:app --host 0.0.0.0 --port ${PORT:-8002}"]
