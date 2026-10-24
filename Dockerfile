# ============================================================
# PDF OCR Backend - Production Dockerfile
# FastAPI + Surya OCR (CPU) — Tamil & English only
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build tools needed to compile native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt ./requirements.txt

# Install all Python dependencies (including Surya OCR)
RUN pip install --default-timeout=200 --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime OS libraries (no tesseract — not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfreetype6 \
        libharfbuzz0b \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Create data dirs and a non-root user first
RUN useradd -m -u 1001 appuser && \
    mkdir -p uploads outputs && \
    chown -R appuser:appuser /app

# Copy application source with appropriate ownership
COPY --chown=appuser:appuser . .

USER appuser

# Models are hosted on separate microservices (PaddleOCR1.5 remote, Embedding remote, LLM remote)
# No local models are pre-downloaded or loaded.


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

EXPOSE 8001

# Exec form so SIGTERM reaches uvicorn cleanly
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001} --workers ${UVICORN_WORKERS:-1} --log-level info"]
