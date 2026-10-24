"""Turn intent router — code, not the LLM.

Resolves what the user wants *before* retrieval and persona selection.
Query expansion for retrieval lives on :class:`ProcessedQuery` (via
:class:`QueryProcessor`); answer prompts always use the original question.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from core.semantic_search import ProcessedQuery, QueryProcessor

logger = logging.getLogger(__name__)

# Re-export for callers that want one import surface
__all__ = [
    "TurnIntent",
    "ProcessedQuery",
    "QueryProcessor",
    "resolve_turn_intent",
    "get_query_language",
    "is_sub_registrar_domain",
    "is_small_talk",
    "select_persona",
    "INTENT_ADDENDA",
]

_query_processor = QueryProcessor()

# ── Small-talk ──────────────────────────────────────────────────────────────

_SMALL_TALK_PHRASES = frozenset({
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
    "good morning", "good afternoon", "good evening", "vanakkam",
    "வணக்கம்", "நன்றி", "bye", "goodbye", "see you",
})

_DOMAIN_KEYWORDS = [
    "registrar", "sub-registrar", "stamp duty", "registration fee", "sale deed",
    "settlement deed", "release deed", "gift deed", "mortgage", "power of attorney",
    "lease deed", "partition deed", "guideline value", "market value",
    "encumbrance certificate", "patta", "chitta", "stamp act", "registration act",
    "standing order", "audit handbook", "succession act", "section 34", "34-c",
    "34c", "encumbrance",
    "பதிவாளர்", "துணைப் பதிவாளர்", "பத்திரம்", "முத்திரை தீர்வை", "வழிகாட்டி மதிப்பு",
    "சந்தை மதிப்பு", "அடமானம்", "பட்டா", "சிட்டா", "வில்லங்க", "பதிவுச் சட்டம்",
    "முத்திரைச் சட்டம்",
]

# Short system-prompt addenda by retrieval intent (not a catalog of sample Qs)
INTENT_ADDENDA: dict[str, str] = {
    "section_lookup": (
        "\nINTENT: Section lookup — quote or paraphrase the relevant section "
        "text from the excerpts precisely. Prefer exact section numbers.\n"
    ),
    "identity_lookup": (
        "\nINTENT: Identity / name lookup — extract the person's name or "
        "identity from headers, contact blocks, or title lines in the excerpts.\n"
    ),
    "summary": (
        "\nINTENT: Summary — give a short, clear overview from the excerpts only. "
        "No invented structure beyond what the documents support.\n"
    ),
    "field_lookup": (
        "\nINTENT: Field extract — answer only the asked field/value from excerpts. "
        "If missing, say so clearly.\n"
    ),
    "survey_lookup": (
        "\nINTENT: Survey / property number — prioritise survey, patta, and plot "
        "identifiers in the excerpts.\n"
    ),
    "boundary_lookup": (
        "\nINTENT: Boundary / schedule — prioritise north/south/east/west and "
        "schedule of property language in the excerpts.\n"
    ),
    "compare": (
        "\nINTENT: Comparison — contrast only what the excerpts state; do not invent differences.\n"
    ),
}


@dataclass(frozen=True)
class TurnIntent:
    """Resolved routing decision for one user turn."""

    retrieval_intent: str
    language: str
    domain: str  # sub_registrar | general
    route: str  # retrieve | small_talk | direct
    need_retrieval: bool
    persona: str  # sub_registrar | general | nova
    processed: ProcessedQuery
    channel: str  # chat | voice

    @property
    def original_query(self) -> str:
        return self.processed.original


def is_small_talk(text: str) -> bool:
    """True for greetings / thanks that should not force document retrieval."""
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
    words = cleaned.split()
    if not words or len(words) > 5:
        return False
    return " ".join(words) in _SMALL_TALK_PHRASES


def is_sub_registrar_domain(
    query: str,
    matches: Optional[Iterable] = None,  # kept for call-site BC; ignored
) -> bool:
    """Query-only domain gate (never use match body text)."""
    del matches  # explicit: body text must not flip persona
    query_lower = (query or "").lower()
    if any(kw in query_lower for kw in _DOMAIN_KEYWORDS):
        return True
    if re.search(r"\bsection\s+\d", query_lower) or re.search(r"\bsec\.?\s*\d", query_lower):
        return True
    return False


def get_query_language(query: str) -> str:
    """Return ``en`` | ``ta`` | ``tanglish`` from the user text."""
    from legal.vocabulary import is_tamil_query

    if is_tamil_query(query or ""):
        return "ta"
    query_lower = (query or "").lower()
    english_words = {
        "what", "is", "this", "doc", "about", "who", "how", "why", "where", "when",
        "show", "list", "get", "find", "search", "section", "page", "document", "the",
        "of", "in", "to", "for", "with", "on", "at", "by", "from", "an", "and", "are",
        "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
        "scanned", "name", "means", "mean",
    }
    words = re.findall(r"\b\w+\b", query_lower)
    if not words:
        return "en"
    english_hits = sum(1 for w in words if w in english_words)
    tanglish_words = {
        "ithu", "pathu", "sollu", "enna", "ennanu", "iruku", "la", "ku", "pathina",
        "kidaikuma", "pannunga", "panna", "eppadi", "yethu", "solla", "theriyum",
        "theriyala", "kedaikatha", "pannala", "paththi", "pathi", "solunga", "sollunga",
    }
    tanglish_hits = sum(1 for w in words if w in tanglish_words)
    if tanglish_hits > english_hits:
        return "tanglish"
    return "en"


def select_persona(*, channel: str, domain: str) -> str:
    """Persona matrix: voice always Nova; chat follows domain."""
    if channel == "voice":
        return "nova"
    return "sub_registrar" if domain == "sub_registrar" else "general"


def resolve_turn_intent(
    query: str,
    *,
    channel: str = "chat",
    has_doc_scope: bool = False,
) -> TurnIntent:
    """Deterministic multi-signal router. No LLM calls."""
    channel = (channel or "chat").strip().lower()
    if channel not in {"chat", "voice"}:
        channel = "chat"

    processed = _query_processor.process(query or "")
    language = get_query_language(processed.original or query or "")
    domain = (
        "sub_registrar"
        if is_sub_registrar_domain(processed.original or query or "")
        else "general"
    )
    persona = select_persona(channel=channel, domain=domain)

    small = is_small_talk(processed.original or query or "")
    if small and not has_doc_scope:
        route = "small_talk"
        need_retrieval = False
    elif small and has_doc_scope:
        # Greeting while a doc is open — answer casually without RAG
        route = "small_talk"
        need_retrieval = False
    else:
        route = "retrieve"
        need_retrieval = True

    intent = TurnIntent(
        retrieval_intent=processed.intent,
        language=language,
        domain=domain,
        route=route,
        need_retrieval=need_retrieval,
        persona=persona,
        processed=processed,
        channel=channel,
    )
    logger.debug(
        "intent resolved channel=%s retrieval_intent=%s domain=%s route=%s "
        "need_retrieval=%s persona=%s lang=%s",
        channel,
        intent.retrieval_intent,
        domain,
        route,
        need_retrieval,
        persona,
        language,
    )
    return intent


def expand_for_retrieve(intent: TurnIntent) -> Sequence[str]:
    """Retrieval-only variants (never pass these as the answer 'question')."""
    return list(intent.processed.variants or [])


def intent_system_addendum(retrieval_intent: str) -> str:
    return INTENT_ADDENDA.get(retrieval_intent or "", "")
