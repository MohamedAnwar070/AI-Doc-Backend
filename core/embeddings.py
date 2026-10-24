"""
Embedding engine — generates float vector embeddings for text chunks.

Provider priority (``EMBEDDING_RUNTIME``):

  * ``in_process`` — sentence-transformers in this process (preferred for latency)
  * ``remote``     — OpenAI-compatible HTTP embedding service
  * ``auto``       — in-process when the model is loadable, else remote / HF

Production voice latency: remote embedding serialises ~7 s/call. In-process
MiniLM/BGE-M3 encodes short queries in single-digit milliseconds. Always prefer
``aget_embeddings`` (batch) over N× ``aget_embedding``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, Sequence

from config import settings

logger = logging.getLogger(__name__)

# Serialize heavy encode work across concurrent requests (CPU ST is not free).
_encode_lock = asyncio.Lock()
_remote_encode_lock = threading.Lock()


def compute_text_hash(text: str, model_name: str) -> str:
    """Compute SHA-256 hash of (model_name + text) for embedding cache key."""
    key = f"{model_name}:{text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def _acquire_remote_encode_lock() -> None:
    """Acquire the cross-event-loop remote lock without blocking or leaking it."""
    while not _remote_encode_lock.acquire(blocking=False):
        await asyncio.sleep(0.05)


def _resolve_local_model_id() -> str:
    """HF id or filesystem path for the in-process SentenceTransformer."""
    path = (getattr(settings, "embedding_local_model_path", None) or "").strip()
    if path:
        return path
    return (
        getattr(settings, "ai_model_embedding_model", None)
        or getattr(settings, "embedding_model_model", None)
        or "BAAI/bge-m3"
    ).strip()


def _expected_dim_hint(model_id: str) -> Optional[int]:
    lower = model_id.lower().replace("\\", "/")
    if "minilm" in lower or "all-minilm" in lower:
        return 384
    if "bge-m3" in lower or "bge_m3" in lower:
        return 1024
    if "e5-base" in lower or "multilingual-e5-base" in lower:
        return 768
    return None


class EmbeddingEngine:
    """Generates float-vector embeddings via in-process ST and/or remote HTTP."""

    def __init__(self) -> None:
        self.client = None  # remote client (BasicAuth / HF)
        self.model_name: str = ""
        self.backend: str = "none"  # in_process | remote | hf | none
        self.dimension: int = 0
        self._local_model = None
        self._local_model_id: str = ""
        self._device: str = getattr(settings, "embedding_device", "cpu") or "cpu"
        self._mem_cache: Dict[str, List[float]] = {}
        self._max_mem_cache_size: int = 5000
        self._setup()


    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        runtime = (getattr(settings, "embedding_runtime", "auto") or "auto").strip().lower()
        if runtime not in {"auto", "in_process", "remote"}:
            logger.warning("Unknown EMBEDDING_RUNTIME=%r — using auto", runtime)
            runtime = "auto"

        if runtime in {"auto", "in_process"}:
            if self._try_setup_local(required=(runtime == "in_process")):
                return
            if runtime == "in_process":
                logger.error(
                    "EMBEDDING_RUNTIME=in_process but local model failed to configure. "
                    "Set EMBEDDING_LOCAL_MODEL_PATH or install sentence-transformers."
                )
                return

        # remote / auto fallback
        if self._try_setup_remote():
            return

        logger.warning(
            "EmbeddingEngine: NO backend configured. "
            "Set EMBEDDING_LOCAL_MODEL_PATH / EMBEDDING_RUNTIME=in_process "
            "or EMBEDDING_MODEL_URL for remote."
        )

    def _try_setup_local(self, required: bool = False) -> bool:
        model_id = _resolve_local_model_id()
        # For auto: only claim local if path exists or HF id is set and ST importable.
        if not required:
            # Prefer local when path exists on disk (deployment mount).
            if os.path.isdir(model_id) or os.path.isfile(model_id):
                pass
            elif getattr(settings, "embedding_runtime", "auto") == "auto":
                # If remote is configured and local path is missing, skip eager local
                # so cold auto deploys without the model still work over HTTP.
                remote_url = getattr(settings, "embedding_model_url", "").strip()
                if remote_url and not os.path.exists(model_id):
                    logger.info(
                        "EmbeddingEngine: local model path %r not present; "
                        "auto will use remote (set EMBEDDING_RUNTIME=in_process to force).",
                        model_id,
                    )
                    return False

        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as exc:
            msg = "sentence-transformers not installed"
            if required:
                logger.error("%s: %s", msg, exc)
            else:
                logger.warning("%s — cannot use in-process embeddings", msg)
            return False

        self._local_model_id = model_id
        self.model_name = model_id
        self.backend = "in_process"
        # Defer actual load to preload() / first encode (startup can schedule it).
        logger.info(
            "EmbeddingEngine ▶ in-process (lazy load)\n"
            "  model   : %s\n"
            "  device  : %s\n"
            "  dim hint: %s",
            model_id,
            self._device,
            _expected_dim_hint(model_id) or "unknown",
        )
        return True

    def _try_setup_remote(self) -> bool:
        local_url = getattr(settings, "embedding_model_url", "").strip()
        local_auth = getattr(settings, "embedding_model_auth", "").strip()
        local_model = getattr(settings, "embedding_model_model", "BAAI/bge-m3")

        if local_url:
            from core.hf_inference_client import BasicAuthLlmClient

            self.client = BasicAuthLlmClient(
                base_url=local_url,
                model=local_model,
                auth=local_auth or "dummy:dummy",
            )
            self.model_name = local_model
            self.backend = "remote"
            logger.info(
                "EmbeddingEngine ▶ remote HTTP\n"
                "  endpoint : %s/embeddings\n"
                "  model    : %s",
                local_url.rstrip("/"),
                local_model,
            )
            return True

        switch_ai = settings.switch_ai
        if switch_ai == "ai_model":
            url = settings.active_ai_model_url()
            auth = settings.active_ai_model_auth()
            model = (
                getattr(settings, "ai_model_embedding_model", "").strip()
                or settings.active_ai_model_model()
                or "local-model"
            )
            if url:
                from core.hf_inference_client import BasicAuthLlmClient

                self.client = BasicAuthLlmClient(base_url=url, model=model, auth=auth)
                self.model_name = model
                self.backend = "remote"
                logger.info(
                    "EmbeddingEngine ▶ chat-server embeddings\n"
                    "  endpoint : %s/embeddings\n"
                    "  model    : %s",
                    url.rstrip("/"),
                    model,
                )
                return True

        hf_key = settings.hf_inference_api_key.strip()
        hf_model = settings.hf_inference_model.strip()
        hf_url = settings.hf_inference_base_url.strip()
        if hf_key and hf_model:
            from core.hf_inference_client import HFInferenceLlmClient

            self.client = HFInferenceLlmClient(
                model=hf_model, api_key=hf_key, base_url=hf_url
            )
            self.model_name = hf_model
            self.backend = "hf"
            logger.warning("EmbeddingEngine ▶ HuggingFace fallback model=%s", hf_model)
            return True

        return False

    # ── Local model load ───────────────────────────────────────────────────

    def _ensure_local_model(self):
        if self._local_model is not None:
            return self._local_model
        if self.backend != "in_process":
            return None

        from sentence_transformers import SentenceTransformer

        model_id = self._local_model_id or _resolve_local_model_id()
        t0 = time.perf_counter()
        logger.info("Loading SentenceTransformer %r on %s …", model_id, self._device)
        try:
            self._local_model = SentenceTransformer(model_id, device=self._device)
        except Exception:
            # Retry without explicit device (some ST builds disagree on device kw).
            self._local_model = SentenceTransformer(model_id)
        load_ms = (time.perf_counter() - t0) * 1000
        try:
            dim = int(self._local_model.get_sentence_embedding_dimension())
        except Exception:
            dim = 0
        self.dimension = dim
        self.model_name = model_id
        logger.info(
            "SentenceTransformer ready model=%s dim=%s load_ms=%.0f",
            model_id,
            dim or "?",
            load_ms,
        )
        return self._local_model

    def _encode_sync(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._ensure_local_model()
        if model is None or not texts:
            return [[] for _ in texts]

        # BGE-M3 / e5: normalize embeddings for cosine similarity.
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if vectors is None:
            return [[] for _ in texts]

        import numpy as np

        arr = np.asarray(vectors)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.dimension <= 0 and arr.size:
            self.dimension = int(arr.shape[-1])
        return [row.astype(float).tolist() for row in arr]

    # ── Public API ─────────────────────────────────────────────────────────

    async def preload(self) -> None:
        """Load the in-process model (if configured) and run a warm encode."""
        if self.backend != "in_process":
            logger.info("Embedding preload skipped (backend=%s)", self.backend)
            return
        t0 = time.perf_counter()
        await asyncio.to_thread(self._ensure_local_model)
        try:
            await self.aget_embeddings(["warmup"])
        except Exception as exc:
            logger.warning("Embedding warm encode failed: %s", exc)
        logger.info(
            "Embedding preload done backend=%s model=%s dim=%s elapsed_ms=%.0f",
            self.backend,
            self.model_name,
            self.dimension or "?",
            (time.perf_counter() - t0) * 1000,
        )

    async def aget_embeddings(self, texts: List[str], db: Any = None) -> List[List[float]]:
        """Batch-embed texts with memory and MongoDB vector caching."""
        if not texts:
            return []

        cleaned: List[str] = []
        empty_idx: List[int] = []
        for i, t in enumerate(texts):
            if not t or not str(t).strip():
                cleaned.append("")
                empty_idx.append(i)
            else:
                cleaned.append(str(t)[:30000])

        t0 = time.perf_counter()
        out: List[List[float]] = [[] for _ in cleaned]

        non_empty = [(i, cleaned[i]) for i in range(len(cleaned)) if i not in empty_idx]
        if not non_empty:
            return out

        cache_enabled = getattr(settings, "embedding_cache_enabled", True)
        if cache_enabled and db is None:
            try:
                from db.mongo import get_db
                db = get_db()
            except Exception:
                db = None

        hashes: Dict[int, str] = {}
        if cache_enabled:
            for idx, text in non_empty:
                h = compute_text_hash(text, self.model_name)
                hashes[idx] = h
                if h in self._mem_cache:
                    out[idx] = self._mem_cache[h]

            # Check MongoDB cache for in-memory misses
            db_missing_indices = [idx for idx, _ in non_empty if not out[idx]]
            if db_missing_indices and db is not None:
                missing_hashes = [hashes[idx] for idx in db_missing_indices]
                try:
                    from db.doc_repo import get_cached_embeddings
                    cached_db = await get_cached_embeddings(db, missing_hashes, self.model_name)
                    for idx in db_missing_indices:
                        h = hashes[idx]
                        if h in cached_db:
                            vec = cached_db[h]
                            out[idx] = vec
                            if len(self._mem_cache) < self._max_mem_cache_size:
                                self._mem_cache[h] = vec
                except Exception as cache_exc:
                    logger.warning("MongoDB embedding cache lookup failed: %s", cache_exc)

        # Identify items that STILL need model inference
        inference_items = [(idx, text) for idx, text in non_empty if not out[idx]]
        if inference_items:
            indices = [i for i, _ in inference_items]
            payloads = [t for _, t in inference_items]

            try:
                if self.backend == "in_process":
                    async with _encode_lock:
                        vectors = await asyncio.to_thread(self._encode_sync, payloads)
                    for i, vec in zip(indices, vectors):
                        out[i] = vec or []
                elif self.client is not None and hasattr(self.client, "get_embeddings"):
                    await _acquire_remote_encode_lock()
                    try:
                        vectors = await self.client.get_embeddings(payloads)
                    finally:
                        _remote_encode_lock.release()
                    for i, vec in zip(indices, vectors):
                        out[i] = list(vec) if vec else []
                elif self.client is not None:
                    for i, text in zip(indices, payloads):
                        try:
                            emb = await self.client.get_embedding(text)
                            out[i] = list(emb) if emb else []
                        except Exception as exc:
                            logger.error("Single embedding failed: %s", exc)
                            out[i] = []
                else:
                    logger.warning("aget_embeddings: no backend available")
            except Exception as exc:
                logger.error(
                    "Batch embedding failed (backend=%s model=%s type=%s): %r",
                    self.backend,
                    self.model_name,
                    type(exc).__name__,
                    exc,
                )
                if self.backend != "in_process" and self.client is not None:
                    short = [t[:8000] for t in payloads]
                    try:
                        await _acquire_remote_encode_lock()
                        try:
                            if hasattr(self.client, "get_embeddings"):
                                vectors = await self.client.get_embeddings(short)
                            else:
                                vectors = []
                                for t in short:
                                    vectors.append(await self.client.get_embedding(t))
                        finally:
                            _remote_encode_lock.release()
                        for i, vec in zip(indices, vectors):
                            out[i] = list(vec) if vec else []
                    except Exception as retry_exc:
                        logger.error(
                            "Batch embedding retry failed (type=%s): %r",
                            type(retry_exc).__name__,
                            retry_exc,
                        )

            # Store newly generated embeddings into cache
            if cache_enabled:
                new_records = []
                for idx, text in inference_items:
                    vec = out[idx]
                    if vec:
                        h = hashes[idx]
                        if len(self._mem_cache) < self._max_mem_cache_size:
                            self._mem_cache[h] = vec
                        new_records.append({
                            "hash": h,
                            "model_name": self.model_name,
                            "embedding": vec,
                        })
                if new_records and db is not None:
                    try:
                        from db.doc_repo import save_cached_embeddings
                        asyncio.create_task(save_cached_embeddings(db, new_records))
                    except Exception as save_exc:
                        logger.warning("Async save_cached_embeddings failed to schedule: %s", save_exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = sum(1 for v in out if v)
        if out and out[0]:
            self.dimension = self.dimension or len(out[0])
        logger.info(
            "stage=embed backend=%s n=%d ok=%d (inferred=%d) dim=%s duration_ms=%.1f",
            self.backend,
            len(texts),
            ok,
            len(inference_items),
            self.dimension or (len(out[0]) if out and out[0] else "?"),
            elapsed_ms,
        )
        return out

    async def aget_embedding(self, text: str) -> List[float]:
        """Generate a float-vector embedding asynchronously (single text)."""
        if not text or not str(text).strip():
            return []
        results = await self.aget_embeddings([text])
        return results[0] if results else []

    def get_embedding(self, text: str) -> List[float]:
        """Synchronous wrapper for background threads (not for use under a running loop)."""
        if not text or not str(text).strip():
            return []
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Cannot block the event loop; schedule via a fresh loop in a worker.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(lambda: asyncio.run(self.aget_embedding(text)))
                try:
                    return fut.result(timeout=120)
                except Exception as exc:
                    logger.error("get_embedding thread failed: %s", exc)
                    return []

        try:
            return asyncio.run(self.aget_embedding(text))
        except Exception as exc:
            logger.error("get_embedding failed: %s", exc)
            return []


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine() -> EmbeddingEngine:
    """Return the shared EmbeddingEngine singleton."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine


def reset_embedding_engine() -> None:
    """Force re-initialisation of the engine (tests / config reload)."""
    global _engine
    _engine = None
