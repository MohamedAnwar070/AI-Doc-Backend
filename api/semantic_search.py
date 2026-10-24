"""Chat semantic-search endpoints — /api/semantic-search, /api/search-suggest.

This module is the **presentation layer** for the chat modality. All
retrieval primitives live in :mod:`core.retrieval`; voice and chat both
call the same :func:`core.retrieval.retrieve` function. Only the
prompt persona (SubRegistrar) and the context-text rendering differ
from the voice modality.

The wire format (``SemanticSearchResponse``, ``MatchResponse``,
``SearchInfo``) is byte-for-byte compatible with the previous version
so the frontend is unaffected.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.context import format_chat_context
from core.llm import get_ai_client
from core.retrieval import (
    MIN_CONTEXT_SCORE,
    MatchItem,
    RetrievalResult,
    RetrievalScope,
    retrieve,
)
from db.mongo import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["semantic_search"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models (unchanged wire format)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query: str
    job_id: Optional[str] = None
    doc_id: Optional[str] = None
    category: Optional[str] = None
    top_k: int = 5
    use_llm: bool = True
    max_tokens: int = 512
    single_doc_only: bool = True
    max_context_chars: int = 6000


class MatchResponse(BaseModel):
    score: float
    doc_id: str
    file_path: str
    chunk_id: int
    text: str
    filename: Optional[str] = None
    page: Optional[int] = None
    section_label: Optional[str] = None
    source: Optional[str] = None


class SearchInfo(BaseModel):
    retrieval_mode: str
    indexed_results: bool
    section_query_refs: List[str] = Field(default_factory=list)
    and_terms: List[str] = Field(default_factory=list)
    and_match_count: int = 0
    exact_section_match_found: bool = False
    answer_confidence: str = "unknown"
    scope: str = "single_doc"
    chunks_evaluated: int = 0
    index_health: Optional[str] = None
    fallback_used: bool = False
    retrieval_stages: List[str] = Field(default_factory=list)
    expected_chunk_count: int = 0
    saved_chunk_count: int = 0
    embedded_chunk_count: int = 0
    query_intent: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    matches: List[MatchResponse]
    search_info: Optional[SearchInfo] = None


class SearchSuggestResponse(BaseModel):
    doc_id: str
    filename: str
    text: str
    page: Optional[int] = None
    section_label: Optional[str] = None
    matched_terms: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-answer path (chat-specific)
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_answer(
    *,
    query: str,
    matches: List[MatchItem],
    max_context_chars: int,
    max_tokens: int,
) -> Optional[str]:
    """Call the unified LLM client with a SubRegistrar persona prompt.

    Returns the LLM's answer with the deterministic citation block
    appended, or ``None`` if generation was skipped / failed. On
    hard failure returns an honest error marker (never silently empty).
    """
    if not matches:
        return None

    from config import settings
    from legal.prompts import SubRegistrarPromptBuilder
    from legal.vocabulary import ensure_tamil_terms

    # Local (basic-auth) LLMs sometimes count tokens by UTF-8 bytes
    # instead of characters. Pick the cost function accordingly so the
    # context never overshoots the model's window.
    is_local = settings.switch_ai == "ai_model"
    budget = max(500, min(int(max_context_chars), 20_000))
    if is_local:
        budget = min(budget, settings.active_ai_model_max_context_chars())

    context_text = format_chat_context(matches, max_chars=budget, is_local=is_local)

    prompt_builder = SubRegistrarPromptBuilder(
        user_query=query,
        matches=list(matches),
    )
    system_prompt = prompt_builder.system_prompt()

    user_content = (
        "Document excerpts:\n\n"
        + context_text
        + "\n\nCRITICAL RULE: Look at the language and script of the User Question below. You MUST reply in the EXACT SAME language and script. If the query is in English, reply in English. If it contains Tamil script, reply in Tamil script. If it is in Tanglish (Tamil words written in English/Latin alphabet, e.g., 'ithu ennanu paathu sollu'), reply in Tanglish (Tamil in English script).\n\n"
        + "User Question: " + query
    )

    try:
        client = get_ai_client()
        raw_answer = await client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_content,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.error("semantic-search LLM call failed: %s", exc, exc_info=True)
        return (
            f"⚠️ The AI model could not generate an answer (error: {type(exc).__name__}). "
            "Relevant document excerpts were found — please check the matches below."
        )

    # If the model refused despite strong evidence in the top chunks, recover
    # with a short grounded answer (or a one-shot stricter re-ask).
    raw_answer = await _recover_false_refusal(
        query=query,
        raw_answer=raw_answer or "",
        matches=matches,
        system_prompt=system_prompt,
        context_text=context_text,
        max_tokens=max_tokens,
    )

    # Apply department Tamil vocabulary (only rewrites Tamil-script answers).
    finalised = ensure_tamil_terms(raw_answer, only_when_tamil=True)
    # Append the deterministic citation block (SubRegistrar finalise).
    return prompt_builder.finalise(finalised)


_REFUSAL_MARKERS = (
    "this information is not available",
    "please contact the sub-registrar",
    "இந்த தகவல் கிடைக்கவில்லை",
    "இந்த தகவல் வழங்கப்பட்ட ஆவணத்தில் இல்லை",
    "itha pathina thagaval",
    "intha thagaval intha aavanathil illai",
)


def _is_refusal_answer(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return True
    return any(m in lower for m in _REFUSAL_MARKERS)


def _query_evidence_in_matches(query: str, matches: List[MatchItem]) -> Optional[str]:
    """Return a short evidence snippet if query terms clearly hit match text."""
    terms = [
        t for t in re.findall(r"\w+", (query or "").lower())
        if len(t) >= 3 and t not in {
            "what", "who", "how", "why", "where", "when", "the", "this",
            "that", "with", "from", "about", "please", "tell", "mean", "means",
        }
    ]
    if not terms:
        return None
    for m in matches[:5]:
        text = m.text or ""
        lower = text.lower()
        hits = sum(1 for t in terms if t in lower)
        if hits >= max(1, (len(terms) + 1) // 2):
            # Prefer the sentence/line containing the strongest term
            for term in terms:
                idx = lower.find(term)
                if idx >= 0:
                    start = max(0, idx - 40)
                    end = min(len(text), idx + 120)
                    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
                    if snippet:
                        return snippet
            return re.sub(r"\s+", " ", text[:180]).strip()
    return None


async def _recover_false_refusal(
    *,
    query: str,
    raw_answer: str,
    matches: List[MatchItem],
    system_prompt: str,
    context_text: str,
    max_tokens: int,
) -> str:
    """When LLM refuses but excerpts clearly answer, re-ask once or extract."""
    if not _is_refusal_answer(raw_answer):
        return raw_answer

    evidence = _query_evidence_in_matches(query, matches)
    if not evidence:
        return raw_answer

    logger.info(
        "semantic-search: false refusal suspected; evidence=%r",
        evidence[:120],
    )
    retry_user = (
        "Document excerpts:\n\n"
        + context_text
        + "\n\nThe excerpts DO contain the answer. Quote or paraphrase the "
        "relevant line(s) only. Do NOT say the information is unavailable.\n\n"
        + f"User Question: {query}"
    )
    try:
        client = get_ai_client()
        retry = await client.generate_response(
            system_prompt=system_prompt,
            user_prompt=retry_user,
            max_tokens=max(128, min(max_tokens, 256)),
        )
        if retry and not _is_refusal_answer(retry):
            return retry.strip()
    except Exception as exc:
        logger.warning("false-refusal retry failed: %s", exc)

    # Deterministic last resort — surface the evidence line
    return f"According to the document: {evidence}"


def _no_match_message(language: str) -> str:
    """Honest 'not found' message in the user's language."""
    if language == "ta":
        return "இந்த தகவல் கிடைக்கவில்லை. துணைப் பதிவாளரை தொடர்பு கொள்ளவும்."
    if language == "tanglish":
        return "Itha pathina thagaval kidaikkavillai. Thunai Pathivalarai thodarbu kollavum."
    return "This information is not available — please contact the Sub-Registrar."


def _enforce_single_doc(
    relevant_matches: List[MatchItem],
    *,
    requested_doc_id: Optional[str],
    scope_label: str,
    single_doc_only: bool,
) -> List[MatchItem]:
    """Apply the chat-specific single-doc scope filter.

    Three rules (in order):
      1. If ``requested_doc_id`` is set, restrict to that document.
      2. If ``single_doc_only`` is True and the search ran across all
         documents, keep only the highest-scoring document's chunks.
      3. Otherwise pass through.
    """
    if requested_doc_id:
        return [m for m in relevant_matches if m.doc_id == requested_doc_id]
    if single_doc_only and scope_label == "all_docs" and relevant_matches:
        best_doc = max(relevant_matches, key=lambda m: m.score).doc_id
        return [m for m in relevant_matches if m.doc_id == best_doc]
    return relevant_matches


# ─────────────────────────────────────────────────────────────────────────────
# Search-suggest endpoint (independent of retrieval — uses MongoDB $text)
# ─────────────────────────────────────────────────────────────────────────────

def _query_terms(query: str) -> List[str]:
    cleaned = query.replace("+", " ")
    return [term for term in re.findall(r"\w+", cleaned.lower()) if len(term) >= 2]


def _preview_for_query(text: str, query: str, radius: int = 90) -> str:
    if not text:
        return ""
    lower_text = text.lower()
    terms = _query_terms(query)
    hit_positions = [lower_text.find(term) for term in terms if lower_text.find(term) >= 0]
    if not hit_positions:
        return text[: radius * 2].strip()
    hit = min(hit_positions)
    start = max(0, hit - radius)
    end = min(len(text), hit + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _matched_terms(texts: List[str], query: str) -> List[str]:
    haystack = " ".join(t for t in texts if t).lower()
    return [term for term in _query_terms(query) if term in haystack][:5]


@router.get("/search-suggest", response_model=List[SearchSuggestResponse])
async def search_suggest(
    q: str = Query(..., min_length=2),
    job_id: Optional[str] = None,
    limit: int = Query(5, ge=1, le=10),
) -> List[SearchSuggestResponse]:
    """Sub-millisecond MongoDB ``$text`` lookup for the search-as-you-type UI.

    Independent of :func:`core.retrieval.retrieve` — this endpoint
    uses MongoDB's native text index so it can answer keystroke-driven
    suggestions without round-tripping through the embedding model.
    """
    query = q.strip()
    if not query:
        return []

    db = get_db()
    mongo_filter: dict = {"job_id": job_id} if job_id else {}

    suggestions: List[dict] = []
    using_documents = False

    # 1. $text on chunks
    try:
        cursor = (
            db.chunks.find(
                {**mongo_filter, "$text": {"$search": query}},
                {
                    "score": {"$meta": "textScore"},
                    "job_id": 1, "filename": 1, "text": 1, "metadata": 1,
                },
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit * 3)
        )
        suggestions = await cursor.to_list(length=limit * 3)
    except Exception as exc:
        logger.warning("search-suggest text query fallback: %s", exc)

    # 2. $text on documents
    if not suggestions:
        try:
            using_documents = True
            cursor = (
                db.documents.find(
                    {**mongo_filter, "$text": {"$search": query}},
                    {
                        "score": {"$meta": "textScore"},
                        "job_id": 1, "filename": 1, "content": 1, "metadata": 1,
                    },
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit * 3)
            )
            suggestions = await cursor.to_list(length=limit * 3)
        except Exception as exc:
            logger.warning("search-suggest document text query fallback: %s", exc)

    # 3. Regex fallback on chunks
    if not suggestions:
        using_documents = False
        regex = re.escape(query)
        regex_filter = {
            **mongo_filter,
            "$or": [
                {"filename": {"$regex": regex, "$options": "i"}},
                {"text": {"$regex": regex, "$options": "i"}},
                {"metadata.section_label": {"$regex": regex, "$options": "i"}},
            ],
        }
        cursor = db.chunks.find(
            regex_filter, {"job_id": 1, "filename": 1, "text": 1, "metadata": 1}
        ).limit(limit * 3)
        suggestions = await cursor.to_list(length=limit * 3)

    # 4. Regex fallback on documents
    if not suggestions:
        using_documents = True
        regex = re.escape(query)
        regex_filter = {
            **mongo_filter,
            "$or": [
                {"filename": {"$regex": regex, "$options": "i"}},
                {"content": {"$regex": regex, "$options": "i"}},
            ],
        }
        cursor = db.documents.find(
            regex_filter, {"job_id": 1, "filename": 1, "content": 1, "metadata": 1}
        ).limit(limit * 3)
        suggestions = await cursor.to_list(length=limit * 3)

    seen: set = set()
    response: List[SearchSuggestResponse] = []
    for item in suggestions:
        metadata = dict(item.get("metadata") or {})
        filename = (
            item.get("filename") or metadata.get("filename") or item.get("job_id", "")
        )
        source_text = (
            item.get("content", "") if using_documents else item.get("text", "")
        )
        preview = _preview_for_query(source_text, query)
        key = item.get("job_id", "")
        if key in seen:
            continue
        seen.add(key)
        response.append(
            SearchSuggestResponse(
                doc_id=item.get("job_id", ""),
                filename=filename,
                text=preview,
                page=metadata.get("page"),
                section_label=metadata.get("section_label"),
                matched_terms=_matched_terms(
                    [filename, source_text, metadata.get("section_label", "")],
                    query,
                ),
            )
        )
        if len(response) >= limit:
            break

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Main semantic-search endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(req: SemanticSearchRequest) -> SemanticSearchResponse:
    """Chat adapter over the shared turn pipeline (intent → retrieve → answer)."""
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = max(1, min(int(req.top_k), 20))
    from config import settings

    if getattr(settings, "search_use_turn_pipeline", True):
        from core.turn_pipeline import TurnRequest, run_answer_turn, to_semantic_search_info

        turn = await run_answer_turn(
            TurnRequest(
                query=q,
                channel="chat",
                doc_id=req.doc_id,
                job_id=req.job_id,
                category=req.category,
                top_k=top_k,
                use_llm=req.use_llm,
                max_tokens=req.max_tokens,
                max_context_chars=req.max_context_chars,
                single_doc_only=req.single_doc_only,
            )
        )
        match_responses = [
            MatchResponse(**m.to_wire_dict()) for m in turn.matches
        ]
        info = to_semantic_search_info(turn)
        allowed = getattr(SearchInfo, "model_fields", None) or getattr(
            SearchInfo, "__fields__", {}
        )
        return SemanticSearchResponse(
            query=q,
            answer=turn.answer,
            matches=match_responses,
            search_info=SearchInfo(**{k: v for k, v in info.items() if k in allowed}),
        )

    # ── Legacy path (flag off) ─────────────────────────────────────────
    scope = RetrievalScope(
        doc_id=req.doc_id,
        job_id=req.job_id,
        category=req.category,
        single_doc_only=req.single_doc_only,
    )
    result: RetrievalResult = await retrieve(q, scope=scope, top_k=top_k)
    match_responses = [MatchResponse(**m.to_wire_dict()) for m in result.matches]
    top_score = match_responses[0].score if match_responses else 0.0
    answer: Optional[str] = None
    answer_confidence = result.confidence

    if result.matches and req.use_llm:
        from core.answer_pack import (
            generate_grounded_answer,
            assemble_answer_pack,
            select_matches_for_answer,
            not_found_message,
        )
        from core.intent import resolve_turn_intent

        intent = resolve_turn_intent(
            q,
            channel="chat",
            has_doc_scope=bool(req.doc_id or req.job_id or req.category),
        )
        selected, soft_gated = select_matches_for_answer(
            result,
            intent,
            modality="chat",
            single_doc_only=req.single_doc_only,
            requested_doc_id=req.doc_id or req.job_id,
        )
        if selected:
            pack = assemble_answer_pack(
                intent=intent,
                result=result,
                selected=selected,
                soft_gated=soft_gated,
                max_context_chars=req.max_context_chars,
            )
            pack = await generate_grounded_answer(
                pack, max_tokens=req.max_tokens
            )
            answer = pack.final_text
        else:
            answer = not_found_message(intent.language, modality="chat")
            answer_confidence = "low"
    elif result.matches and not req.use_llm:
        if result.section_refs and not result.exact_section_match_found:
            answer = (
                "I found related text, but no strong exact section heading "
                "match in the indexed excerpts yet."
            )
        elif top_score < float(settings.search_answer_min_score):
            answer = (
                "I found low-confidence OCR matches only. "
                "Try a more specific query or re-index the document."
            )

    return SemanticSearchResponse(
        query=q,
        answer=answer,
        matches=match_responses,
        search_info=SearchInfo(
            retrieval_mode=result.retrieval_mode,
            indexed_results=result.retrieval_mode == "mongodb",
            section_query_refs=result.section_refs,
            and_terms=result.and_terms,
            and_match_count=result.and_match_count,
            exact_section_match_found=result.exact_section_match_found,
            answer_confidence=answer_confidence,
            scope=result.scope_label,
            chunks_evaluated=result.chunks_evaluated,
            index_health=result.index_health,
            fallback_used=result.fallback_used,
            retrieval_stages=result.retrieval_stages,
            expected_chunk_count=result.expected_chunk_count,
            saved_chunk_count=result.saved_chunk_count,
            embedded_chunk_count=result.embedded_chunk_count,
            query_intent=result.query_intent,
        ),
    )
