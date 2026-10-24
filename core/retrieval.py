"""Unified retrieval pipeline for chat and voice.

This module owns **all** retrieval semantics. Both
``api/semantic_search.py`` (chat) and ``voice/conversation_engine.py``
(voice) call :func:`retrieve` and never re-implement any of the steps
inside it. That way a query improvement (multi-query variants, BM25,
RRF re-ranking, relevance gating) instantly benefits every modality.

What this module owns
---------------------
* Query preprocessing (cleaning, variants, key terms, intent)
* Multi-query variant embedding (one call per variant)
* TF-IDF lexical scoring (with sklearn fallback to keyword score)
* BM25 scoring
* Hybrid scoring (vector + lexical + BM25 + section boost)
* Stage-1 broad retrieval
* Stage-2 RRF re-ranking with ``compute_rerank_score``
* Relevance gating via :data:`MIN_CONTEXT_SCORE`

What this module deliberately does NOT own
-----------------------------------------
* LLM calls — caller picks the persona (chat → SubRegistrar, voice →
  Nova) and the LLM client factory.
* Context-text formatting — chat wants ``[Source i] doc_id=X``,
  voice wants ``[Page X — filename]``. :mod:`core.context` handles
  both shapes against the data this module returns.
* Prompt construction — :mod:`legal.prompts`.
* Conversation history — caller-specific (chat persists to MongoDB,
  voice keeps in WebSocket session).

Wire format compatibility
-------------------------
The dataclasses in this module are the *internal* contract. Both
``SemanticSearchResponse`` (chat API) and ``VoiceAssistantResult``
(voice WebSocket) project their own view of the same data. Field
names match the existing API response so the frontend is not affected.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from core.semantic_search import (
    MIN_CONTEXT_SCORE,
    QueryProcessor,
    _bm25_score,
    _build_df_map,
    _exact_phrase_score,
    _keyword_score,
    compute_confidence,
    compute_rerank_score,
    extract_legal_field_hits,
    extract_query_section_refs,
    hybrid_score,
    identity_chunk_boost,
    parse_and_terms,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


# ── Scope and result dataclasses ────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalScope:
    """Which documents the search runs against.

    Priority order is resolved inside :func:`retrieve`:

        ``doc_id`` or ``job_id``  >  ``category``  >  all docs

    Both ``doc_id`` and ``job_id`` are accepted because the chat API
    uses ``doc_id`` in its public schema while the worker's per-page
    pipeline produces ``job_id``. They mean the same thing.

    Attributes:
        doc_id: Limit search to one document (job_id) by primary id.
        job_id: Same as ``doc_id``. Either one is sufficient.
        category: Limit search to one document category (e.g.
            ``"court_orders"``, ``"medical"``).
        single_doc_only: When ``True`` (the chat default), if the query
            matches across multiple documents, restrict the answer to
            the highest-scoring document. Set ``False`` for cross-doc
            synthesis (e.g. "compare X across all uploaded reports").
    """

    doc_id: Optional[str] = None
    job_id: Optional[str] = None
    category: Optional[str] = None
    single_doc_only: bool = True

    @property
    def effective_doc_id(self) -> Optional[str]:
        """First non-null identifier in priority order."""
        return self.doc_id or self.job_id or None

    @property
    def scope_label(self) -> str:
        """Stable string for logging / response metadata."""
        if self.effective_doc_id:
            return "single_doc"
        if self.category:
            return "category"
        return "all_docs"


@dataclass
class MatchItem:
    """One retrieved chunk, normalised across both endpoints.

    This is the *single source of truth* for what a chunk looks like
    after retrieval. Both the chat endpoint and the voice engine read
    these fields directly — they never reach back into the raw Mongo
    document.
    """

    score: float
    doc_id: str
    chunk_id: int
    text: str
    filename: Optional[str] = None
    file_path: Optional[str] = ""
    page: Optional[int] = None
    section_label: Optional[str] = None
    source: Optional[str] = "mongodb"
    metadata: dict = field(default_factory=dict)

    def to_wire_dict(self) -> dict:
        """Render for the chat API response.

        Field names match the historical ``MatchResponse`` so the
        frontend renderer is unaffected.
        """
        return {
            "score": self.score,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "filename": self.filename,
            "page": self.page,
            "section_label": self.section_label,
            "source": self.source,
        }


@dataclass
class RetrievalResult:
    """Everything a caller needs to render a chat or voice answer.

    Both ``matches`` and ``relevant_matches`` are sorted by score
    descending. ``matches`` is the full top-k; ``relevant_matches`` is
    the same list filtered to ``score >= min_score``.
    """

    matches: List[MatchItem]
    relevant_matches: List[MatchItem]
    chunks_evaluated: int
    retrieval_mode: str  # "mongodb" | "fallback"
    and_terms: List[str]
    section_refs: List[str]
    exact_section_match_found: bool
    confidence: str  # "high" | "medium" | "low" | "unknown"
    scope_label: str  # "single_doc" | "category" | "all_docs"
    and_match_count: int = 0
    query_intent: str = "keyword"
    index_health: str = "unknown"
    fallback_used: bool = False
    retrieval_stages: List[str] = field(default_factory=list)
    expected_chunk_count: int = 0
    saved_chunk_count: int = 0
    embedded_chunk_count: int = 0


# ── Public entry point ──────────────────────────────────────────────────────

# Module-level query processor is stateless and reusable across calls
_query_processor = QueryProcessor()


async def retrieve(
    query: str,
    *,
    scope: RetrievalScope,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> RetrievalResult:
    """Run the full retrieval pipeline against MongoDB (or file fallback).

    Args:
        query: Raw user text. Will be cleaned, normalised, and expanded
            into query variants internally.
        scope: Which documents to search (single / category / all).
        top_k: Maximum number of matches to return after re-ranking.
            Capped at 20 to keep prompt sizes bounded.
        min_score: Minimum hybrid score for a chunk to count as
            relevant. Defaults to :data:`MIN_CONTEXT_SCORE` from
            :mod:`core.semantic_search`. Set to ``0.0`` to disable.

    Returns:
        A :class:`RetrievalResult` carrying every signal the chat and
        voice paths need: matches, relevant subset, scope metadata,
        confidence, section refs, AND-term hits. Callers project this
        into their own response shape.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")

    top_k = max(1, min(int(top_k), 20))
    threshold = MIN_CONTEXT_SCORE if min_score is None else float(min_score)

    # ── 0. Preprocess query ─────────────────────────────────────────────
    pq = _query_processor.process(q)
    logger.info(
        "retrieve: query=%r → cleaned=%r, key_terms=%s, variants=%d, intent=%s, scope=%s",
        q,
        pq.cleaned,
        pq.key_terms,
        len(pq.variants),
        pq.intent,
        scope.scope_label,
    )

    and_terms = parse_and_terms(q)
    section_refs = extract_query_section_refs(q)

    # ── 1. Load chunks based on scope ───────────────────────────────────
    from db.doc_repo import get_all_chunks, get_chunks_by_category
    from db.mongo import get_db

    db = get_db()
    effective_doc_id = scope.effective_doc_id

    if effective_doc_id:
        db_chunks_raw = await get_all_chunks(db, job_id=effective_doc_id)
        logger.info(
            "retrieve: SINGLE_DOC mode — doc_id=%s, %d chunks loaded",
            effective_doc_id,
            len(db_chunks_raw),
        )
        if not db_chunks_raw:
            db_chunks_raw = await _try_build_missing_single_doc_index(
                effective_doc_id
            )
    elif scope.category:
        db_chunks_raw = await get_chunks_by_category(db, category=scope.category)
        logger.info(
            "retrieve: CATEGORY mode — category=%s, %d chunks loaded",
            scope.category,
            len(db_chunks_raw),
        )
    else:
        db_chunks_raw = await get_all_chunks(db, job_id=None)
        logger.info(
            "retrieve: ALL_DOCS mode — %d chunks loaded",
            len(db_chunks_raw),
        )

    # ── 2. MongoDB hybrid path (all chunks; vectors only where present) ───
    if db_chunks_raw:
        raw_count = len(db_chunks_raw)
        db_chunks_raw, prefilter_stage = _prefilter_all_doc_candidates(
            pq, db_chunks_raw, scope=scope
        )
        if prefilter_stage:
            logger.info(
                "retrieve: %s reduced chunks from %d to %d",
                prefilter_stage,
                raw_count,
                len(db_chunks_raw),
            )
        scored = await _score_chunks_mongodb(
            db_chunks_raw, pq, top_k=top_k, threshold=threshold,
            and_terms=and_terms, section_refs=section_refs,
            scope=scope,
            retrieval_stages=["mongo_chunks"] + ([prefilter_stage] if prefilter_stage else []),
        )
        return scored

    # ── 3. File-based TF-IDF fallback (no MongoDB chunks) ───────────────
    logger.info(
        "retrieve: no embedded chunks in MongoDB for scope=%s — "
        "falling back to file-based TF-IDF search.",
        scope.scope_label,
    )
    scored = _score_chunks_fallback(
        q, top_k=top_k, threshold=threshold, scope=scope,
        and_terms=and_terms, section_refs=section_refs,
    )
    return scored


# ── MongoDB scoring path ────────────────────────────────────────────────────

async def _try_build_missing_single_doc_index(job_id: str) -> List[dict]:
    """Build a missing per-document index once before falling back.

    Background OCR threads schedule indexing asynchronously. If that task
    fails or is missed, the first chat/voice query should not be stuck in
    permanent fallback mode. This helper is intentionally limited to
    single-document scope so a broad search cannot accidentally backfill
    the whole corpus inside a user request.
    """
    from core.job_manager import get_job_manager
    from config import settings
    from db.doc_repo import get_all_chunks
    from db.mongo import get_db

    manager = get_job_manager()
    status = manager.progress_tracker.get_summary(job_id)
    if status.get("status") != "completed":
        logger.info(
            "retrieve: doc_id=%s has no chunks and status=%s; skipping on-demand indexing",
            job_id,
            status.get("status"),
        )
        return []

    if status.get("search_index_status") == "indexing":
        started_raw = status.get("search_index_started_at")
        stale_seconds = int(getattr(settings, "search_index_stale_seconds", 300))
        wait_seconds = int(getattr(settings, "search_index_wait_seconds", 45))
        age_seconds = 0.0
        if started_raw:
            try:
                age_seconds = (datetime.now() - datetime.fromisoformat(started_raw)).total_seconds()
            except Exception:
                age_seconds = 0.0
        if age_seconds < stale_seconds:
            # Background index is running (e.g. BGE-M3 first load). Poll briefly
            # so the first user question does not permanently miss embeddings.
            logger.info(
                "retrieve: doc_id=%s indexing in progress (age=%.1fs); "
                "waiting up to %ds for chunks",
                job_id,
                age_seconds,
                wait_seconds,
            )
            deadline = time.time() + max(0, wait_seconds)
            while time.time() < deadline:
                await asyncio.sleep(1.0)
                chunks = await get_all_chunks(get_db(), job_id=job_id)
                if chunks:
                    logger.info(
                        "retrieve: doc_id=%s chunks became available mid-index (%d)",
                        job_id,
                        len(chunks),
                    )
                    return chunks
            logger.info(
                "retrieve: doc_id=%s still no chunks after wait; falling through",
                job_id,
            )
            return []
        logger.warning(
            "retrieve: doc_id=%s indexing is stale (age=%.1fs, no chunks); rebuilding on demand",
            job_id,
            age_seconds,
        )

    try:
        logger.info(
            "retrieve: doc_id=%s has no embedded chunks; building index on demand",
            job_id,
        )
        outcome = await manager.index_job(job_id, force=False)
        logger.info(
            "retrieve: on-demand index result for doc_id=%s: indexed=%s chunks=%s",
            job_id,
            outcome.get("indexed"),
            outcome.get("chunk_count"),
        )
    except Exception as exc:
        logger.error(
            "retrieve: on-demand indexing failed for doc_id=%s: %s",
            job_id,
            exc,
            exc_info=True,
        )
        return []

    return await get_all_chunks(get_db(), job_id=job_id)


async def _score_chunks_mongodb(
    db_chunks_raw: List[dict],
    pq,  # ProcessedQuery — type from core.semantic_search
    *,
    top_k: int,
    threshold: float,
    and_terms: List[str],
    section_refs: List[str],
    scope: RetrievalScope,
    retrieval_stages: List[str],
) -> RetrievalResult:
    """Score every chunk against every query variant, then RRF re-rank."""
    from core.embeddings import get_embedding_engine

    engine = get_embedding_engine()

    # 2a. Multi-query variant embeddings (max score across variants wins).
    #     One batch call — critical when remote serialises concurrency, and
    #     free for in-process sentence-transformers.
    variant_texts = list(pq.variants)
    embed_targets = variant_texts + [pq.cleaned]  # cleaned query embedded last
    try:
        embed_raws = await engine.aget_embeddings(embed_targets)
    except Exception as emb_exc:
        logger.error("retrieve: batch embedding failed: %s", emb_exc)
        embed_raws = [[] for _ in embed_targets]

    if len(embed_raws) < len(embed_targets):
        embed_raws = list(embed_raws) + [[] for _ in range(len(embed_targets) - len(embed_raws))]

    query_embeddings: List[np.ndarray] = []
    for emb_raw in embed_raws[: len(variant_texts)]:
        if emb_raw:
            query_embeddings.append(np.array(emb_raw, dtype=np.float32))

    original_emb_raw = embed_raws[-1] if embed_raws else []
    if original_emb_raw:
        original_emb = np.array(original_emb_raw, dtype=np.float32)
        if not query_embeddings:
            query_embeddings.append(original_emb)
        elif original_emb.size > 0:
            already_covered = any(
                np.allclose(original_emb, existing, atol=1e-4)
                for existing in query_embeddings
                if existing.size == original_emb.size
            )
            if not already_covered:
                query_embeddings.append(original_emb)

    # Dimension mismatch vs corpus → silent vector skip. Surface it loudly.
    if query_embeddings and db_chunks_raw:
        q_dim = int(query_embeddings[0].size)
        sample_dims = []
        for doc in db_chunks_raw[:20]:
            emb = doc.get("embedding") or []
            if emb:
                sample_dims.append(len(emb))
                if len(sample_dims) >= 3:
                    break
        if sample_dims and any(d != q_dim for d in sample_dims):
            logger.warning(
                "retrieve: query embedding dim=%d differs from chunk dims=%s — "
                "vector cosine will be skipped for mismatched chunks. "
                "Re-index documents after changing EMBEDDING_LOCAL_MODEL_PATH / model.",
                q_dim,
                sample_dims,
            )

    logger.info(
        "retrieve: embedded %d query variants (batch n=%d)",
        len(query_embeddings),
        len(embed_targets),
    )

    # 2b. Lexical scores (TF-IDF with sklearn → keyword-score fallback)
    texts = [doc.get("text", "") for doc in db_chunks_raw]
    df_map, n_docs = _build_df_map(texts)
    lexical_scores = _compute_tfidf_scores(pq.cleaned, texts)

    # 2c. BM25 average-doc-length normalisation
    avg_doc_len = sum(len(t.split()) for t in texts) / max(len(texts), 1)

    # 2d. Stage 1 — score every chunk
    stage1_pool_size = min(max(top_k * 8, 30), len(db_chunks_raw))
    scored: List[dict] = []
    embedded_chunk_count = 0

    for idx, doc in enumerate(db_chunks_raw):
        doc_emb = np.array(doc.get("embedding", []), dtype=np.float32)
        if doc_emb.size > 0:
            embedded_chunk_count += 1

        # Multi-query vector score: max cosine across all variants
        vector_score = 0.0
        if doc_emb.size > 0 and query_embeddings:
            for q_emb in query_embeddings:
                if q_emb.size == doc_emb.size:
                    norm = np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)
                    sim = float(np.dot(q_emb, doc_emb) / (norm + 1e-9))
                    vector_score = max(vector_score, sim)

        chunk_bm25 = _bm25_score(
            pq.cleaned,
            doc.get("text", ""),
            avg_doc_len,
            df_map=df_map,
            n_docs=n_docs,
        )

        metadata = dict(doc.get("metadata") or {})
        filename = doc.get("filename") or metadata.get("filename")
        metadata["filename"] = filename
        text = doc.get("text", "")
        field_score = _field_lookup_score(pq, text, metadata)
        exact_score = max(
            _exact_phrase_score(pq.original, text),
            _exact_phrase_score(pq.cleaned, text),
        )
        chunk_type_boost = _chunk_type_boost(pq.intent, metadata)

        final_score = hybrid_score(
            query=pq.cleaned,
            text=text,
            metadata=metadata,
            lexical_score=lexical_scores[idx] if idx < len(lexical_scores) else 0.0,
            vector_score=vector_score,
            bm25_score_val=chunk_bm25,
            query_word_count=pq.word_count,
        )
        final_score += field_score + (0.30 * exact_score) + chunk_type_boost
        if pq.intent == "identity_lookup":
            final_score += identity_chunk_boost(text, metadata)
        scored.append(
            {
                "idx": idx,
                "score": final_score,
                "vector_score": vector_score,
                "bm25_score": chunk_bm25,
                "lexical_score": lexical_scores[idx] if idx < len(lexical_scores) else 0.0,
                "doc_id": doc.get("job_id", ""),
                "chunk_id": doc.get("chunk_id", 0),
                "text": text,
                "file_path": doc.get("file_path")
                or f"db://{doc.get('job_id', '')}/chunk/{doc.get('chunk_id', 0)}",
                "filename": filename,
                "page": metadata.get("page"),
                "section_label": metadata.get("section_label"),
                "source": metadata.get("source") or "mongodb",
                "metadata": metadata,
                "field_score": field_score,
                "exact_score": exact_score,
            }
        )

    # AND-term sort boost
    if and_terms:
        scored.sort(
            key=lambda x: (
                _and_hit_count(x["text"], and_terms) == len(and_terms),
                x["score"],
            ),
            reverse=True,
        )
    else:
        scored.sort(key=lambda x: x["score"], reverse=True)

    # 2e. Stage 2 — RRF re-rank the top candidates
    broad_set = scored[:stage1_pool_size]
    if len(broad_set) > top_k:
        by_vector = sorted(broad_set, key=lambda x: x["vector_score"], reverse=True)
        by_bm25 = sorted(broad_set, key=lambda x: x["bm25_score"], reverse=True)
        by_tfidf = sorted(broad_set, key=lambda x: x["lexical_score"], reverse=True)
        by_field = sorted(broad_set, key=lambda x: (x["field_score"], x["exact_score"]), reverse=True)

        rrf_input_vector = [m["idx"] for m in by_vector]
        rrf_input_bm25 = [m["idx"] for m in by_bm25]
        rrf_input_tfidf = [m["idx"] for m in by_tfidf]
        rrf_input_field = [m["idx"] for m in by_field]

        rrf_results = reciprocal_rank_fusion(
            rrf_input_vector, rrf_input_bm25, rrf_input_tfidf, rrf_input_field, k=60
        )
        rrf_score_map = {chunk_idx: rrf_s for chunk_idx, rrf_s in rrf_results}

        for m in broad_set:
            rerank = compute_rerank_score(
                query_terms=pq.key_terms,
                text=m["text"],
                vector_score=m["vector_score"],
                bm25=m["bm25_score"],
                lexical=m["lexical_score"],
                metadata=m.get("metadata"),
            )
            rrf_s = rrf_score_map.get(m["idx"], 0.0)
            max_rrf = max((s for _, s in rrf_results), default=1.0)
            m["score"] = 0.60 * rerank + 0.40 * (rrf_s / max(max_rrf, 1e-9))
            m["score"] += 0.50 * m.get("field_score", 0.0) + 0.20 * m.get("exact_score", 0.0)

        broad_set.sort(key=lambda x: x["score"], reverse=True)

    # Bound scores to ~[0, 1] before MIN_CONTEXT_SCORE gating (F9).
    # Relative max-norm keeps ranking identical; absolute gate becomes meaningful.
    _normalize_score_list(broad_set)

    index_meta = _index_meta(db_chunks_raw)
    retrieval_mode = "hybrid" if embedded_chunk_count else "lexical"
    stages = list(dict.fromkeys(
        retrieval_stages
        + ["query_embedding" if query_embeddings else "query_embedding_unavailable"]
        + ["vector_score" if embedded_chunk_count and query_embeddings else "lexical_score"]
        + ["bm25", "field_window", "rrf_rerank"]
    ))

    return _build_result(
        broad_set[:top_k], db_chunks_raw, threshold=threshold,
        and_terms=and_terms, section_refs=section_refs,
        scope=scope, retrieval_mode=retrieval_mode,
        query_intent=pq.intent,
        index_health=index_meta["index_health"],
        fallback_used=index_meta["index_health"] in {"degraded", "lexical_only", "failed"},
        retrieval_stages=stages,
        expected_chunk_count=index_meta["expected_chunk_count"],
        saved_chunk_count=index_meta["saved_chunk_count"],
        embedded_chunk_count=index_meta["embedded_chunk_count"],
    )


# ── File-based fallback ─────────────────────────────────────────────────────

def _score_chunks_fallback(
    query: str,
    *,
    top_k: int,
    threshold: float,
    scope: RetrievalScope,
    and_terms: List[str],
    section_refs: List[str],
) -> RetrievalResult:
    """File-based TF-IDF fallback when MongoDB has no embeddings."""
    from config import settings
    from core.job_manager import get_job_manager
    from core.semantic_search import build_corpus, search_chunks

    pq = _query_processor.process(query)
    job_manager = get_job_manager()
    effective_doc_id = scope.effective_doc_id
    chunks = build_corpus(
        storage=job_manager.storage,
        chunk_size=getattr(settings, "search_chunk_size", 1200),
        overlap=getattr(settings, "search_chunk_overlap", 150),
        job_id=effective_doc_id,
        doc_id=effective_doc_id,
    )
    # Search with cleaned + original; identity queries also need header heuristics.
    search_q = pq.cleaned or query
    matches = search_chunks(search_q, chunks, top_k=max(top_k * 3, 15))
    if pq.original and pq.original.lower() != search_q.lower():
        alt = search_chunks(pq.original, chunks, top_k=max(top_k * 3, 15))
        by_id = {(m.chunk.doc_id, m.chunk.chunk_id): m for m in matches}
        for m in alt:
            key = (m.chunk.doc_id, m.chunk.chunk_id)
            if key not in by_id or m.score > by_id[key].score:
                by_id[key] = m
        matches = sorted(by_id.values(), key=lambda m: m.score, reverse=True)

    scored = []
    for m in matches:
        meta = dict(m.chunk.metadata or {})
        score = float(m.score)
        if pq.intent == "identity_lookup":
            score += identity_chunk_boost(m.chunk.text, meta)
        scored.append(
            {
                "score": score,
                "doc_id": m.chunk.doc_id,
                "chunk_id": m.chunk.chunk_id,
                "text": m.chunk.text,
                "file_path": m.chunk.file_path,
                "filename": meta.get("filename"),
                "page": meta.get("page"),
                "section_label": meta.get("section_label"),
                "source": meta.get("source") or "fallback",
                "metadata": meta,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    _normalize_score_list(scored)
    scored = scored[:top_k]

    # Single-doc fallback: never leave the user with zero relevant context when
    # we clearly have the only document's chunks (threshold is for multi-doc).
    effective_threshold = threshold
    if scope.effective_doc_id and scored:
        effective_threshold = min(threshold, 0.12)

    return _build_result(
        scored, db_chunks_raw=None, threshold=effective_threshold,
        and_terms=and_terms, section_refs=section_refs,
        scope=scope, retrieval_mode="fallback",
        query_intent=pq.intent,
        index_health="fallback",
        fallback_used=True,
        retrieval_stages=["file_corpus", "tfidf", "bm25", "identity_boost"],
        expected_chunk_count=len(chunks),
        saved_chunk_count=len(chunks),
        embedded_chunk_count=0,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_score_list(items: List[dict], score_key: str = "score") -> None:
    """In-place max-normalise scores into [0, 1] so MIN_CONTEXT_SCORE is meaningful.

    Ranking order is preserved. If all scores are <= 0, leave them unchanged.
    """
    if not items:
        return
    max_s = max(float(m.get(score_key, 0.0) or 0.0) for m in items)
    if max_s <= 1e-12:
        return
    # Only scale down when unbounded additives pushed scores past 1.0
    if max_s <= 1.0:
        for m in items:
            s = float(m.get(score_key, 0.0) or 0.0)
            m[score_key] = max(0.0, min(1.0, s))
        return
    for m in items:
        s = float(m.get(score_key, 0.0) or 0.0)
        m[score_key] = max(0.0, min(1.0, s / max_s))


def _compute_tfidf_scores(query: str, texts: List[str]) -> List[float]:
    """TF-IDF cosine similarity per chunk, with sklearn failure fallback.

    Mirrors the behaviour of the chat endpoint's private helper so the
    ranking is byte-for-byte identical to the legacy path.
    """
    if not texts:
        return []
    try:
        vectorizer = TfidfVectorizer(
            stop_words=None, max_features=50_000, ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(texts)
        q_vec = vectorizer.transform([query])
        return sk_cosine(q_vec, matrix).flatten().tolist()
    except Exception:
        return [_keyword_score(query, t) for t in texts]


def _field_lookup_score(pq, text: str, metadata: dict) -> float:
    """Boost exact legal/property field matches and close term windows."""
    if pq.intent not in {"field_lookup", "boundary_lookup", "survey_lookup"}:
        return 0.0
    lower = (text or "").lower()
    query_terms = [t for t in pq.key_terms if len(t) >= 2]
    if not query_terms:
        return 0.0

    coverage = sum(1 for term in query_terms if term in lower) / max(1, len(query_terms))
    field_hits = set(metadata.get("field_hits") or extract_legal_field_hits(text))
    score = 0.35 * coverage
    if field_hits:
        score += 0.20

    if pq.intent == "boundary_lookup" and field_hits & {"north", "south", "east", "west"}:
        score += 0.30
    if pq.intent == "survey_lookup" and any("survey" in hit for hit in field_hits):
        score += 0.30
    if pq.intent == "boundary_lookup" and any("survey" in hit for hit in field_hits):
        score += 0.22

    positions = []
    tokens = re.findall(r"\w+", lower)
    for idx, token in enumerate(tokens):
        if any(term == token or term in token for term in query_terms):
            positions.append(idx)
    if len(positions) >= 2 and max(positions) - min(positions) <= 60:
        score += 0.25

    return min(score, 1.25)


def _chunk_type_boost(intent: str, metadata: dict) -> float:
    chunk_type = metadata.get("chunk_type")
    if intent == "summary" and chunk_type == "page":
        return 0.25
    if intent in {"field_lookup", "boundary_lookup", "survey_lookup"}:
        if chunk_type in {"fact_window", "schedule", "table"}:
            return 0.25
    return 0.0


def _prefilter_all_doc_candidates(pq, chunks: List[dict], *, scope: RetrievalScope) -> tuple[List[dict], str]:
    """For all-doc corpora, rank documents first to avoid noisy full-corpus scoring."""
    if scope.scope_label != "all_docs" or len(chunks) <= 3000:
        return chunks, ""

    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.get("job_id", ""), []).append(chunk)

    ranked_docs: list[tuple[str, float]] = []
    for doc_id, doc_chunks in grouped.items():
        best = 0.0
        for chunk in doc_chunks[:80]:
            metadata = dict(chunk.get("metadata") or {})
            text = chunk.get("text", "")
            score = max(
                _keyword_score(pq.cleaned, text),
                _bm25_score(pq.cleaned, text),
                _filename_doc_score(pq.cleaned, chunk.get("filename") or metadata.get("filename")),
            )
            score += _field_lookup_score(pq, text, metadata)
            best = max(best, score)
        ranked_docs.append((doc_id, best))

    ranked_docs.sort(key=lambda item: item[1], reverse=True)
    keep = {doc_id for doc_id, score in ranked_docs[:30] if score > 0}
    if not keep:
        keep = {doc_id for doc_id, _ in ranked_docs[:10]}
    return [chunk for chunk in chunks if chunk.get("job_id", "") in keep], "doc_prefilter"


def _filename_doc_score(query: str, filename: Optional[str]) -> float:
    return _keyword_score(query, filename or "")


def _index_meta(chunks: List[dict]) -> dict:
    saved = len(chunks)
    embedded = sum(1 for c in chunks if c.get("embedding"))
    expected_values = [
        int((c.get("metadata") or {}).get("expected_chunk_count") or 0)
        for c in chunks
        if (c.get("metadata") or {}).get("expected_chunk_count")
    ]
    expected = max(expected_values) if expected_values else saved
    if saved == 0:
        health = "missing"
    elif embedded == 0:
        health = "lexical_only"
    elif embedded < saved:
        health = "degraded"
    else:
        health = "healthy"
    return {
        "index_health": health,
        "expected_chunk_count": expected,
        "saved_chunk_count": saved,
        "embedded_chunk_count": embedded,
    }


def _and_hit_count(text: str, terms: Iterable[str]) -> int:
    """Number of AND-separated query terms that occur in ``text``."""
    if not text or not terms:
        return 0
    haystack = text.lower()
    return sum(1 for term in terms if term.lower() in haystack)


def _build_result(
    scored_top: List[dict],
    db_chunks_raw: Optional[List[dict]],
    *,
    threshold: float,
    and_terms: List[str],
    section_refs: List[str],
    scope: RetrievalScope,
    retrieval_mode: str,
    query_intent: str,
    index_health: str,
    fallback_used: bool,
    retrieval_stages: List[str],
    expected_chunk_count: int,
    saved_chunk_count: int,
    embedded_chunk_count: int,
) -> RetrievalResult:
    """Wrap scored top-k into a :class:`RetrievalResult`."""
    matches = [_to_match_item(m) for m in scored_top]

    relevant = [m for m in matches if m.score >= threshold]

    exact_section_match_found = False
    if section_refs:
        for m in matches:
            haystacks = [(m.section_label or "").lower(), m.text.lower()]
            if any(
                ref.lower() in haystack
                for ref in section_refs
                for haystack in haystacks
            ):
                exact_section_match_found = True
                break

    confidence = compute_confidence(
        [m.score for m in matches],
        section_refs,
        exact_section_match_found,
    )

    and_match_count = (
        sum(1 for m in matches if _and_hit_count(m.text, and_terms) == len(and_terms))
        if and_terms
        else 0
    )

    return RetrievalResult(
        matches=matches,
        relevant_matches=relevant,
        chunks_evaluated=len(db_chunks_raw) if db_chunks_raw is not None else 0,
        retrieval_mode=retrieval_mode,
        and_terms=list(and_terms),
        section_refs=list(section_refs),
        exact_section_match_found=exact_section_match_found,
        confidence=confidence,
        scope_label=scope.scope_label,
        and_match_count=and_match_count,
        query_intent=query_intent,
        index_health=index_health,
        fallback_used=fallback_used,
        retrieval_stages=retrieval_stages,
        expected_chunk_count=expected_chunk_count,
        saved_chunk_count=saved_chunk_count,
        embedded_chunk_count=embedded_chunk_count,
    )


def _to_match_item(m: dict) -> MatchItem:
    """Promote a raw scored-dict into a typed :class:`MatchItem`."""
    return MatchItem(
        score=float(m.get("score", 0.0)),
        doc_id=str(m.get("doc_id", "")),
        chunk_id=int(m.get("chunk_id", 0)),
        text=str(m.get("text", "")),
        filename=m.get("filename"),
        file_path=m.get("file_path", ""),
        page=m.get("page"),
        section_label=m.get("section_label"),
        source=m.get("source") or "mongodb",
        metadata=dict(m.get("metadata") or {}),
    )


__all__ = [
    "RetrievalScope",
    "MatchItem",
    "RetrievalResult",
    "retrieve",
    "MIN_CONTEXT_SCORE",
]
