"""Shared answer brain: intent → retrieve → persona → grounded answer.

Chat HTTP and voice WebSocket adapters both call :func:`run_answer_turn`.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.answer_pack import (
    AnswerPack,
    assemble_answer_pack,
    generate_grounded_answer,
    not_found_message,
    select_matches_for_answer,
)
from core.intent import TurnIntent, resolve_turn_intent
from core.retrieval import MatchItem, RetrievalResult, RetrievalScope, retrieve

logger = logging.getLogger("core.turn_pipeline")


@dataclass
class TurnRequest:
    query: str
    channel: str = "chat"  # chat | voice
    doc_id: Optional[str] = None
    job_id: Optional[str] = None
    category: Optional[str] = None
    top_k: int = 5
    use_llm: bool = True
    max_tokens: int = 512
    max_context_chars: int = 6000
    single_doc_only: bool = True
    history: Optional[List[Dict[str, str]]] = None


@dataclass
class TurnResult:
    query: str
    answer: Optional[str]
    matches: List[MatchItem]
    intent: TurnIntent
    retrieval: Optional[RetrievalResult]
    pack: AnswerPack
    timings_ms: Dict[str, int] = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def _flags():
    from config import settings

    return settings


async def run_answer_turn(req: TurnRequest) -> TurnResult:
    """Human query → intent → retrieve → select → LLM editor → TurnResult."""
    t_total = time.perf_counter()
    timings: Dict[str, int] = {}
    q = (req.query or "").strip()
    channel = (req.channel or "chat").lower()
    has_scope = bool(req.doc_id or req.job_id or req.category)

    t0 = time.perf_counter()
    intent = resolve_turn_intent(
        q, channel=channel, has_doc_scope=has_scope
    )
    timings["intent_ms"] = int((time.perf_counter() - t0) * 1000)

    settings = _flags()
    use_pipeline = True
    if channel == "chat":
        use_pipeline = bool(getattr(settings, "search_use_turn_pipeline", True))
    else:
        use_pipeline = bool(getattr(settings, "voice_use_turn_pipeline", True))
    # Flag is reserved for gradual rollout; pipeline is the only path when called.
    del use_pipeline

    # ── Small-talk short circuit (no retrieval) ─────────────────────────
    if not intent.need_retrieval:
        pack = AnswerPack(
            query_original=intent.original_query or q,
            intent=intent,
            persona=intent.persona,
            final_text=_small_talk_reply(intent),
            confidence="high",
        )
        result = TurnResult(
            query=q,
            answer=pack.final_text,
            matches=[],
            intent=intent,
            retrieval=None,
            pack=pack,
            timings_ms=timings,
        )
        timings["total_ms"] = int((time.perf_counter() - t_total) * 1000)
        result.timings_ms = timings
        _log_turn(result, llm_called=False, soft_gated=False)
        return result

    # ── Retrieve ────────────────────────────────────────────────────────
    scope = RetrievalScope(
        doc_id=req.doc_id,
        job_id=req.job_id,
        category=req.category,
        single_doc_only=req.single_doc_only,
    )
    t1 = time.perf_counter()
    retrieval = await retrieve(q, scope=scope, top_k=req.top_k)
    timings["retrieve_ms"] = int((time.perf_counter() - t1) * 1000)

    soft_gate_enabled = True
    if channel == "voice":
        soft_gate_enabled = bool(
            getattr(settings, "voice_soft_gate_single_doc", True)
        )

    selected, soft_gated = select_matches_for_answer(
        retrieval,
        intent,
        modality=channel,
        single_doc_only=req.single_doc_only,
        requested_doc_id=req.doc_id or req.job_id,
        soft_gate_enabled=soft_gate_enabled,
    )

    pack = assemble_answer_pack(
        intent=intent,
        result=retrieval,
        selected=selected,
        soft_gated=soft_gated,
        max_context_chars=req.max_context_chars,
        history=req.history,
    )

    answer: Optional[str] = pack.final_text
    confidence = retrieval.confidence

    if req.use_llm and selected:
        t2 = time.perf_counter()
        pack = await generate_grounded_answer(
            pack,
            max_tokens=req.max_tokens,
            refusal_recovery=bool(
                getattr(settings, "search_refusal_recovery", True)
            )
            and channel == "chat",
        )
        timings["llm_ms"] = int((time.perf_counter() - t2) * 1000)
        answer = pack.final_text
    elif not selected:
        answer = not_found_message(intent.language, modality=channel)
        pack.final_text = answer
        confidence = "low"
    elif not req.use_llm:
        # Matches only — optional light hints for chat
        answer = None
        if retrieval.section_refs and not retrieval.exact_section_match_found:
            answer = (
                "I found related text, but no strong exact section heading "
                "match in the indexed excerpts yet."
            )
        pack.final_text = answer

    timings["total_ms"] = int((time.perf_counter() - t_total) * 1000)
    turn = TurnResult(
        query=q,
        answer=answer,
        matches=list(retrieval.matches),
        intent=intent,
        retrieval=retrieval,
        pack=pack,
        timings_ms=timings,
    )
    _log_turn(
        turn,
        llm_called=pack.llm_called,
        soft_gated=soft_gated,
    )
    return turn


def _small_talk_reply(intent: TurnIntent) -> str:
    if intent.language == "ta":
        return "வணக்கம்! ஆவணம் பற்றி கேளுங்கள் — உதவத் தயாராக இருக்கிறேன்."
    if intent.language == "tanglish":
        return "Vanakkam! Document pathi ketinga — help panna ready."
    if intent.channel == "voice":
        return "Hi! Ask me anything about your uploaded documents."
    return "Hello! Ask a question about your documents and I will answer from them."


def _log_turn(
    turn: TurnResult,
    *,
    llm_called: bool,
    soft_gated: bool,
) -> None:
    r = turn.retrieval
    top_score = turn.matches[0].score if turn.matches else 0.0
    logger.info(
        "turn_id=%s channel=%s intent=%s lang=%s domain=%s route=%s persona=%s "
        "scope=%s need_retrieval=%s chunks_evaluated=%s top_score=%.3f "
        "soft_gated=%s llm_called=%s refusal_recovery=%s matches=%d "
        "selected=%d retrieve_ms=%s llm_ms=%s total_ms=%s",
        turn.turn_id,
        turn.intent.channel,
        turn.intent.retrieval_intent,
        turn.intent.language,
        turn.intent.domain,
        turn.intent.route,
        turn.intent.persona,
        r.scope_label if r else "none",
        turn.intent.need_retrieval,
        r.chunks_evaluated if r else 0,
        top_score,
        soft_gated,
        llm_called,
        turn.pack.refusal_recovery,
        len(turn.matches),
        len(turn.pack.selected_matches),
        turn.timings_ms.get("retrieve_ms", 0),
        turn.timings_ms.get("llm_ms", 0),
        turn.timings_ms.get("total_ms", 0),
    )


def to_semantic_search_info(turn: TurnResult) -> Dict[str, Any]:
    """Build search_info dict for chat wire format."""
    r = turn.retrieval
    if r is None:
        return {
            "retrieval_mode": "none",
            "indexed_results": False,
            "section_query_refs": [],
            "and_terms": [],
            "and_match_count": 0,
            "exact_section_match_found": False,
            "answer_confidence": turn.pack.confidence or "high",
            "scope": "none",
            "chunks_evaluated": 0,
            "query_intent": turn.intent.retrieval_intent,
        }
    info: Dict[str, Any] = {
        "retrieval_mode": r.retrieval_mode,
        "indexed_results": r.retrieval_mode == "mongodb",
        "section_query_refs": r.section_refs,
        "and_terms": r.and_terms,
        "and_match_count": r.and_match_count,
        "exact_section_match_found": r.exact_section_match_found,
        "answer_confidence": turn.pack.confidence or r.confidence,
        "scope": r.scope_label,
        "chunks_evaluated": r.chunks_evaluated,
        "index_health": r.index_health,
        "fallback_used": r.fallback_used,
        "retrieval_stages": r.retrieval_stages,
        "expected_chunk_count": r.expected_chunk_count,
        "saved_chunk_count": r.saved_chunk_count,
        "embedded_chunk_count": r.embedded_chunk_count,
        "query_intent": r.query_intent or turn.intent.retrieval_intent,
    }
    from config import settings

    if getattr(settings, "search_info_expose_intent", False):
        info["turn_intent"] = turn.intent.retrieval_intent
        info["persona"] = turn.intent.persona
        info["domain"] = turn.intent.domain
    return info
