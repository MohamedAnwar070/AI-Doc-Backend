"""Embedding API — /api/embed/*
Accepts OCR output (job_id + text) and stores embeddings in MongoDB.
Also provides a vector-search endpoint that returns relevant chunks given a query.

Uses the shared :mod:`core.embeddings` engine (in-process BGE-M3 by default)
so this router does not load a second SentenceTransformer instance.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from db.mongo import get_db
from db import embedding_repo
from db.models import EmbeddingChunk, UserPublic
from auth.dependencies import get_current_user
from config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/embed", tags=["embedding"])


def _embedding_model_name() -> str:
    """Logical model id for API responses / logging."""
    from core.embeddings import get_embedding_engine, _resolve_local_model_id

    engine = get_embedding_engine()
    return engine.model_name or _resolve_local_model_id()


async def _embed_texts(texts: List[str]) -> List[List[float]]:
    from core.embeddings import get_embedding_engine

    engine = get_embedding_engine()
    return await engine.aget_embeddings(texts)


def _embed_texts_sync(texts: List[str]) -> List[List[float]]:
    """Sync helper for legacy code paths that are not async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        raise RuntimeError("Use async embed path under a running event loop")
    return asyncio.run(_embed_texts(texts))


def _chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """Split ``text`` into overlapping chunks for embedding.

    Defaults match :pyattr:`config.Settings.search_chunk_size` and
    :pyattr:`config.Settings.search_chunk_overlap` so this fallback path
    produces the same shape of chunk as the main indexer.
    """
    size = chunk_size if chunk_size is not None else settings.search_chunk_size
    step_back = overlap if overlap is not None else settings.search_chunk_overlap
    if size <= 0:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        segment = text[start:end].strip()
        if segment:
            chunks.append(segment)
        if end == len(text):
            break
        start += max(1, size - step_back)
    return chunks


# ── Request/Response models ──────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    job_id: str
    doc_id: str = ""
    text: str


class EmbedResponse(BaseModel):
    job_id: str
    chunks_stored: int


class SearchRequest(BaseModel):
    job_id: str
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    score: float
    text: str
    metadata: dict = {}


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("", response_model=EmbedResponse)
async def embed_text(
    body: EmbedRequest,
    current_user: UserPublic = Depends(get_current_user),
):
    """Chunk and embed OCR text, store in MongoDB."""
    chunks = _chunk_text(body.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty text")

    embeddings = await _embed_texts(chunks)
    if not any(embeddings):
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding backend returned empty vectors. "
                f"Check EMBEDDING_RUNTIME / EMBEDDING_LOCAL_MODEL_PATH "
                f"(model={_embedding_model_name()})."
            ),
        )
    chunk_objs = [
        EmbeddingChunk(
            job_id=body.job_id,
            doc_id=body.doc_id or body.job_id,
            chunk_id=i,
            text=c,
            embedding=embeddings[i] if i < len(embeddings) else [],
        )
        for i, c in enumerate(chunks)
    ]

    db = get_db()
    stored = await embedding_repo.upsert_chunks(db, chunk_objs)
    return EmbedResponse(job_id=body.job_id, chunks_stored=stored)


@router.post("/search", response_model=List[SearchResult])
async def vector_search(
    body: SearchRequest,
    current_user: UserPublic = Depends(get_current_user),
):
    """Return top-k most relevant chunks for a query."""
    vectors = await _embed_texts([body.query])
    q_embedding = vectors[0] if vectors else []
    if not q_embedding:
        raise HTTPException(
            status_code=503,
            detail="Failed to embed query — embedding backend unavailable",
        )
    db = get_db()
    results = await embedding_repo.similarity_search(db, body.job_id, q_embedding, body.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No embeddings found for this job")
    return [SearchResult(**r) for r in results]
