from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
import multiprocessing
from contextlib import asynccontextmanager

# ------------------------------------------------------------------
# Windows multiprocessing guard
# Must be called before any Process/Executor is created.
# 'spawn' is the Windows default but set it explicitly so the inner
# ProcessPoolExecutor inside PDFWorker always works correctly.
# ------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    pass  # already set (harmless in reload mode)

from api import (
    upload_router, status_router, result_router, semantic_search_router,
    auth_router, chats_router, embedding_router, categories_router,
    tags_router, vision_router, layout_router, translation_router,
)
from voice import voice_router
from db.mongo import connect_db, close_db
from db.category_repo import seed_default_categories
from config import settings

# ── Logging configuration ────────────────────────────────────────────────────
# Guard against duplicate handlers (uvicorn reload / double import).
# Only add our StreamHandler if the root logger has none yet.
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

if not logging.root.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.root.addHandler(_handler)
else:
    # Re-apply our format to the existing handler(s) to keep output consistent.
    for _h in logging.root.handlers:
        if not _h.formatter:
            _h.setFormatter(logging.Formatter(_LOG_FORMAT))

logging.root.setLevel(logging.INFO)

# Silence noisy third-party / infrastructure loggers.
for _noisy in (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "uvicorn.access",        # /health, OPTIONS, etc. — still see errors
    "urllib3",
    "urllib3.connectionpool",
    "asyncio",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING if _noisy != "asyncio" else logging.CRITICAL)

logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    import asyncio
    # Startup
    logger.info("=" * 60)
    logger.info("PDF OCR Backend Starting")
    logger.info("=" * 60)
    logger.info(f"OCR Engine: {settings.ocr_engine}")
    logger.info(f"Max Workers: {settings.max_workers}")
    logger.info(f"GPU Enabled: {settings.use_gpu}")
    logger.info(f"Upload Directory: {settings.upload_dir}")
    logger.info(f"Output Directory: {settings.output_dir}")
    logger.info("=" * 60)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Connect MongoDB
    await connect_db()
    logger.info("MongoDB connected")

    # Seed predefined document categories if collection is empty
    from db.mongo import get_db
    await seed_default_categories(get_db())
    logger.info("Document categories ready")
    
    # Initialize embedding engine; optionally warm the in-process model.
    from core.embeddings import get_embedding_engine

    embed_engine = get_embedding_engine()
    logger.info(
        "Embedding backend=%s model=%s path=%s remote_url=%s",
        embed_engine.backend,
        embed_engine.model_name,
        getattr(settings, "embedding_local_model_path", "") or "(none)",
        (getattr(settings, "embedding_model_url", "") or "").strip() or "(disabled)",
    )
    if embed_engine.backend == "remote":
        remote = (getattr(settings, "embedding_model_url", "") or "").strip()
        if remote and ("localhost" in remote or "127.0.0.1" in remote):
            logger.info(
                "Using local embedding microservice at %s (BGE-M3 warm pool).",
                remote,
            )
        else:
            logger.warning(
                "Embeddings use remote HTTP (%s). For lowest latency on this host "
                "prefer EMBEDDING_RUNTIME=in_process or a local embedding_api "
                "on localhost.",
                remote or embed_engine.backend,
            )
    if getattr(settings, "embedding_preload", True):
        try:
            # Background so Mongo/seed already done; first request still waits if needed.
            asyncio.create_task(embed_engine.preload())
            logger.info(
                "Embedding preload scheduled (backend=%s model=%s)",
                embed_engine.backend,
                embed_engine.model_name,
            )
        except Exception as e:
            logger.error("Failed to schedule embedding preload: %s", e)

    # Preload voice models in the background if NEED_VOICE is enabled
    if getattr(settings, "need_voice", False):
        logger.info("NEED_VOICE is enabled. Pre-loading voice models...")
        try:
            from voice.stt_service import preload_model as preload_stt
            from voice.tts_service import preload_model as preload_tts
            
            # Use asyncio.create_task so startup completes instantly while models load in the background
            asyncio.create_task(preload_stt())
            asyncio.create_task(preload_tts())
            logger.info("Voice models preloading task scheduled in the background.")
        except Exception as e:
            logger.error("Failed to schedule voice model preloading: %s", e)

    # Capture the main event loop for background jobs
    from core.job_manager import get_job_manager
    manager = get_job_manager()
    manager.loop = asyncio.get_running_loop()
    manager.start_queue()
    logger.info("JobManager loop captured and durable queue started")

    yield

    # Shutdown
    await close_db()
    logger.info("PDF OCR Backend Shutting Down")


# Create FastAPI app
app = FastAPI(
    title="AI Doc Backend",
    description="Production-ready multi-engine OCR, bilingual RAG, and real-time voice AI backend — Architected & built by Mohamed Anwar", contact={"name": "Mohamed Anwar", "url": "https://github.com/MohamedAnwar070"},
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:7175",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,  # Cache preflight (OPTIONS) for 10 minutes
)

# Include routers
app.include_router(upload_router)
app.include_router(status_router)
app.include_router(result_router)
app.include_router(semantic_search_router)
app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(embedding_router)
app.include_router(categories_router)
app.include_router(tags_router)
app.include_router(vision_router)
app.include_router(layout_router)
app.include_router(translation_router)
app.include_router(voice_router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AI Doc Backend", "author": "Mohamed Anwar (https://github.com/MohamedAnwar070)",
        "version": "1.0.0",
        "status": "running",
        "ocr_engine": settings.ocr_engine,
        "endpoints": {
            "upload": "POST /api/upload",
            "status": "GET /api/status/{job_id}",
            "result": "GET /api/result/{job_id}",
            "result_text": "GET /api/result/{job_id}/text",
            "download": "GET /api/result/{job_id}/download",
            "cancel": "POST /api/cancel/{job_id}",
            "delete": "DELETE /api/delete/{job_id}",
            "vision": "POST /api/vision/analyze"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_engine": settings.ocr_engine,
        "need_voice": settings.need_voice,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    import uvicorn
    import os
    
    workers = int(os.environ.get("UVICORN_WORKERS", "1"))
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=workers,
        log_level="info"
    )
