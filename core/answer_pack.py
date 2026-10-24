"""Grounded answer assembly — match selection, prompts, LLM editor, recovery.

Chat and voice both use this module so answer policy stays one place.
Citations: chat finalises with a footer; voice returns structured sources only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.context import format_chat_context, format_voice_context
from core.intent import TurnIntent, intent_system_addendum
from core.llm import get_ai_client
from core.retrieval import MatchItem, RetrievalResult

logger = logging.getLogger(__name__)

_REFUSAL_MARKERS = (
    "this information is not available",
    "please contact the sub-registrar",
    "இந்த தகவல் கிடைக்கவில்லை",
    "இந்த தகவல் வழங்கப்பட்ட ஆவணத்தில் இல்லை",
    "itha pathina thagaval",
    "intha thagaval intha aavanathil illai",
    "i couldn't find that information",
)


@dataclass
class AnswerPack:
    """Everything needed to produce a grounded answer for one modality."""

    query_original: str
    intent: TurnIntent
    persona: str
    selected_matches: List[MatchItem] = field(default_factory=list)
    context_text: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    confidence: str = "unknown"
    soft_gated: bool = False
    final_text: Optional[str] = None
    sources_wire: List[Dict[str, Any]] = field(default_factory=list)
    refusal_recovery: int = 0
    llm_called: bool = False


def enforce_single_doc(
    matches: Sequence[MatchItem],
    *,
    requested_doc_id: Optional[str],
    scope_label: str,
    single_doc_only: bool,
) -> List[MatchItem]:
    """Restrict matches to one document when the client scoped or requested it."""
    items = list(matches)
    if requested_doc_id:
        return [m for m in items if m.doc_id == requested_doc_id]
    if single_doc_only and scope_label == "all_docs" and items:
        best_doc = max(items, key=lambda m: m.score).doc_id
        return [m for m in items if m.doc_id == best_doc]
    return items


def select_matches_for_answer(
    result: RetrievalResult,
    intent: TurnIntent,
    *,
    modality: str,
    single_doc_only: bool,
    requested_doc_id: Optional[str],
    soft_gate_enabled: bool = True,
) -> Tuple[List[MatchItem], bool]:
    """Prefer relevant_matches; soft-fall back for single-doc / fallback index."""
    del intent  # reserved for future intent-aware selection
    modality = (modality or "chat").lower()
    selected = enforce_single_doc(
        result.relevant_matches,
        requested_doc_id=requested_doc_id,
        scope_label=result.scope_label,
        single_doc_only=single_doc_only,
    )
    soft_gated = False
    if selected or not soft_gate_enabled:
        return selected, soft_gated

    scoped = bool(requested_doc_id) or result.scope_label == "single_doc"
    allow_soft = scoped or result.fallback_used
    # Voice multi-doc: keep hard gate (avoid weak open-corpus grounding)
    if modality == "voice" and not scoped and not result.fallback_used:
        return [], False

    if allow_soft and result.matches:
        selected = enforce_single_doc(
            result.matches,
            requested_doc_id=requested_doc_id,
            scope_label=result.scope_label,
            single_doc_only=single_doc_only,
        )
        soft_gated = bool(selected)
        if soft_gated:
            logger.info(
                "answer_pack soft-gate modality=%s n=%d top_score=%.3f fallback=%s",
                modality,
                len(selected),
                selected[0].score if selected else 0.0,
                result.fallback_used,
            )
    return selected, soft_gated


def not_found_message(language: str, *, modality: str = "chat") -> str:
    if modality == "voice":
        if language == "ta":
            return "மன்னிக்கவும், பதிவேற்றிய ஆவணங்களில் அந்தத் தகவலைக் கண்டறிய முடியவில்லை."
        if language == "tanglish":
            return (
                "Mannikkavum, upload pannina document-la antha thagaval "
                "kandupidikka mudiyala."
            )
        return "I couldn't find that information in the uploaded documents."
    if language == "ta":
        return "இந்த தகவல் கிடைக்கவில்லை. துணைப் பதிவாளரை தொடர்பு கொள்ளவும்."
    if language == "tanglish":
        return "Itha pathina thagaval kidaikkavillai. Thunai Pathivalarai thodarbu kollavum."
    return "This information is not available — please contact the Sub-Registrar."


def _build_system_prompt(intent: TurnIntent, matches: Sequence[MatchItem]) -> str:
    from legal.prompts import (
        NovaPromptBuilder,
        SubRegistrarPromptBuilder,
        _GENERAL_HEADER,
        build_system_prompt,
    )

    addendum = intent_system_addendum(intent.retrieval_intent)
    if intent.persona == "nova":
        # Context is injected separately for Nova when context_text is set
        return ""  # filled in assemble after context known

    if intent.persona == "sub_registrar":
        base = build_system_prompt(language_hint=intent.language)
    else:
        base = _GENERAL_HEADER
        if intent.language == "ta":
            base += (
                "\nLANGUAGE OVERRIDE: The user question contains Tamil script. "
                "You MUST reply ENTIRELY in formal Tamil script (தமிழ்).\n"
            )
        elif intent.language == "tanglish":
            base += (
                "\nLANGUAGE OVERRIDE: The user question is in Tanglish. "
                "You MUST reply ENTIRELY in Tanglish (Latin-script Tamil words).\n"
            )
        else:
            base += (
                "\nLANGUAGE OVERRIDE: The user question is in English. "
                "You MUST reply ENTIRELY in English.\n"
            )
    # Builder still used for citation finalise shape consistency
    _ = SubRegistrarPromptBuilder(
        user_query=intent.original_query,
        matches=list(matches),
    )
    return base + addendum


def assemble_answer_pack(
    *,
    intent: TurnIntent,
    result: RetrievalResult,
    selected: List[MatchItem],
    soft_gated: bool,
    max_context_chars: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AnswerPack:
    """Build context + prompts. Does not call the LLM."""
    from config import settings

    pack = AnswerPack(
        query_original=intent.original_query,
        intent=intent,
        persona=intent.persona,
        selected_matches=list(selected),
        soft_gated=soft_gated,
        confidence=result.confidence,
    )

    if not selected:
        pack.final_text = not_found_message(
            intent.language, modality=intent.channel
        )
        return pack

    is_local = settings.switch_ai == "ai_model"
    budget = max(500, min(int(max_context_chars), 20_000))
    if is_local:
        budget = min(budget, settings.active_ai_model_max_context_chars())

    if intent.channel == "voice":
        context_text, sources = format_voice_context(selected, max_chars=min(budget, 4000))
        pack.context_text = context_text
        pack.sources_wire = sources
        from legal.prompts import NovaPromptBuilder

        builder = NovaPromptBuilder(
            user_query=intent.original_query,
            matches=sources,
            conversation_history=history or [],
            context_text=context_text,
        )
        pack.system_prompt = builder.system_prompt() + intent_system_addendum(
            intent.retrieval_intent
        )
        # Original question only — never expanded retrieval variants
        pack.user_prompt = intent.original_query
    else:
        pack.context_text = format_chat_context(
            selected, max_chars=budget, is_local=is_local
        )
        pack.sources_wire = [m.to_wire_dict() for m in selected]
        pack.system_prompt = _build_system_prompt(intent, selected)
        pack.user_prompt = (
            "Document excerpts:\n\n"
            + pack.context_text
            + "\n\nAnswer ONLY using the excerpts above. "
            "If the answer is present (including headers, footers, stamps, "
            "scanner lines, names, or dates), state it clearly. "
            "Do not invent facts.\n\n"
            f"User Question: {intent.original_query}"
        )
    return pack


def is_refusal_answer(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return True
    return any(m in lower for m in _REFUSAL_MARKERS)


def query_evidence_in_matches(
    query: str, matches: Sequence[MatchItem]
) -> Optional[str]:
    terms = [
        t
        for t in re.findall(r"\w+", (query or "").lower())
        if len(t) >= 3
        and t
        not in {
            "what", "who", "how", "why", "where", "when", "the", "this",
            "that", "with", "from", "about", "please", "tell", "mean", "means",
        }
    ]
    if not terms:
        return None
    for m in list(matches)[:5]:
        text = m.text or ""
        lower = text.lower()
        hits = sum(1 for t in terms if t in lower)
        if hits >= max(1, (len(terms) + 1) // 2):
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


async def recover_false_refusal(
    *,
    query: str,
    raw_answer: str,
    matches: Sequence[MatchItem],
    system_prompt: str,
    context_text: str,
    max_tokens: int,
) -> Tuple[str, int]:
    if not is_refusal_answer(raw_answer):
        return raw_answer, 0
    evidence = query_evidence_in_matches(query, matches)
    if not evidence:
        return raw_answer, 0
    logger.info("answer_pack: false refusal suspected; evidence=%r", evidence[:120])
    retry_user = (
        "Document excerpts:\n\n"
        + context_text
        + "\n\nThe excerpts DO contain the answer. Quote or paraphrase the "
        "relevant line(s) only. Do NOT say the information is unavailable.\n\n"
        f"User Question: {query}"
    )
    try:
        client = get_ai_client()
        retry = await client.generate_response(
            system_prompt=system_prompt,
            user_prompt=retry_user,
            max_tokens=max(128, min(max_tokens, 256)),
        )
        if retry and not is_refusal_answer(retry):
            return retry.strip(), 1
    except Exception as exc:
        logger.warning("false-refusal retry failed: %s", exc)
    return f"According to the document: {evidence}", 1


async def generate_grounded_answer(
    pack: AnswerPack,
    *,
    max_tokens: int,
    refusal_recovery: bool = True,
) -> AnswerPack:
    """Call the LLM as evidence editor; post-process by modality."""
    if pack.final_text is not None and not pack.selected_matches:
        return pack
    if not pack.selected_matches or not pack.system_prompt:
        if pack.final_text is None:
            pack.final_text = not_found_message(
                pack.intent.language, modality=pack.intent.channel
            )
        return pack

    pack.llm_called = True
    try:
        client = get_ai_client()
        raw = await client.generate_response(
            system_prompt=pack.system_prompt,
            user_prompt=pack.user_prompt,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.error("grounded answer LLM failed: %s", exc, exc_info=True)
        pack.final_text = (
            f"⚠️ The AI model could not generate an answer (error: {type(exc).__name__}). "
            "Relevant document excerpts were found — please check the matches below."
            if pack.intent.channel == "chat"
            else not_found_message(pack.intent.language, modality="voice")
        )
        return pack

    raw = raw or ""
    if refusal_recovery and pack.intent.channel == "chat":
        raw, n = await recover_false_refusal(
            query=pack.query_original,
            raw_answer=raw,
            matches=pack.selected_matches,
            system_prompt=pack.system_prompt,
            context_text=pack.context_text,
            max_tokens=max_tokens,
        )
        pack.refusal_recovery = n

    if pack.intent.channel == "voice":
        # Strip tool JSON / markdown for speech
        if _looks_like_tool_json(raw):
            pack.final_text = not_found_message(
                pack.intent.language, modality="voice"
            )
        else:
            pack.final_text = clean_for_voice(raw)
    else:
        from legal.prompts import SubRegistrarPromptBuilder
        from legal.vocabulary import ensure_tamil_terms

        text = ensure_tamil_terms(raw, only_when_tamil=True)
        builder = SubRegistrarPromptBuilder(
            user_query=pack.query_original,
            matches=list(pack.selected_matches),
        )
        pack.final_text = builder.finalise(text)

    return pack


def _looks_like_tool_json(text: str) -> bool:
    t = (text or "").strip()
    return '"tool"' in t and "search_pdf" in t


def clean_for_voice(text: str) -> str:
    """Remove formatting and spoken citation furniture."""
    if not text:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"⚠️|📌|📝|✅|❌|💡|🔍", "", text)
    text = re.sub(
        r"\(\s*Page\s+\d+(?:\s*,\s*Section\s+[^)]+)?\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\(\s*page\s+\d+[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*Page\s+\d+[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()
