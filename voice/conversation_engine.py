"""Voice conversation engine — Nova AI Assistant.

This module is the **voice modality wrapper** for the unified
retrieval pipeline. Both the chat ``/api/semantic-search`` endpoint and
this engine call the **same** :func:`core.retrieval.retrieve`
function — the user-facing constraint is that voice must never have
its own retrieval implementation. Differences between modalities live
in:

* The prompt persona — chat uses ``SubRegistrarPromptBuilder``,
  voice uses ``NovaPromptBuilder``. Both live in
  :mod:`legal.prompts`.
* The context-text rendering — chat wants ``[Source i] doc_id=X``,
  voice wants ``[Page X — filename]``. Both live in :mod:`core.context`.
* The I/O modality — STT → text in, TTS-cleaned text out. That's
  what this module owns.

Two-pass LLM flow
-----------------
1. **Pass 1** — LLM decides if it needs a document search or can
   answer directly (greetings, casual chat, out-of-scope questions).
2. **Pass 2** — if the LLM emitted a ``search_pdf`` tool call, run
   :func:`core.retrieval.retrieve`, render the voice context, and
   call the LLM again with the retrieved context appended to the
   system prompt.

The conversation history (last six messages) is threaded through both
passes so the user can ask "explain that" or "what about the previous
section?".
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.context import format_voice_context
from core.llm import get_ai_client
from core.retrieval import RetrievalScope, retrieve
from legal.prompts import NovaPromptBuilder, get_query_language

logger = logging.getLogger(__name__)


# ── Public response dataclass ───────────────────────────────────────────────

@dataclass
class VoiceAssistantResult:
    """Response payload for realtime voice UI."""

    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)


# ── String safety helper (kept from the original) ──────────────────────────

def _safe_str(s: str) -> str:
    """Safe encoder for terminal logging on Windows to prevent UnicodeEncodeError."""
    if not s:
        return ""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        s.encode(encoding)
        return s
    except Exception:
        return s.encode("ascii", errors="backslashreplace").decode("ascii")


# ── Engine ──────────────────────────────────────────────────────────────────

class VoiceConversationEngine:
    """Orchestrates the Nova voice conversation pipeline.

    For each user utterance:
        1. First LLM pass — decide if search_pdf is needed or answer
           directly.
        2. If tool call: call :func:`core.retrieval.retrieve` (the same
           function the chat endpoint uses), render voice context,
           second LLM pass with context, return response + sources.
        3. Clean the response for voice delivery.
    """

    def __init__(self) -> None:
        self._ai_client = None

    def _get_ai_client(self):
        """Lazy-load the unified LLM client (shared with chat)."""
        if self._ai_client is None:
            self._ai_client = get_ai_client()
        return self._ai_client

    # ── Public API ─────────────────────────────────────────────────────

    async def generate_response(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
        detected_language: str = "en",
    ) -> str:
        """Generate a voice-friendly text response (no sources).

        Convenience wrapper around :meth:`generate_response_with_sources`
        for callers that don't need citation metadata.
        """
        result = await self.generate_response_with_sources(
            user_text=user_text,
            conversation_history=conversation_history,
            doc_id=doc_id,
            category=category,
            detected_language=detected_language,
        )
        return result.text

    async def generate_response_with_sources(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
        detected_language: str = "en",
    ) -> VoiceAssistantResult:
        """Generate a voice-friendly response via the shared turn pipeline.

        When ``voice_use_turn_pipeline`` is enabled (default), document-scoped
        and small-talk turns share chat's intent→retrieve→answer brain. Unscoped
        free-form turns still use Nova pass-1 tool routing as a fallback.
        """
        conversation_history = conversation_history or []
        detected_language = get_query_language(user_text)

        from config import settings

        if getattr(settings, "voice_use_turn_pipeline", True):
            from core.intent import is_small_talk, resolve_turn_intent
            from core.turn_pipeline import TurnRequest, run_answer_turn

            has_scope = bool(doc_id or category)
            intent = resolve_turn_intent(
                user_text, channel="voice", has_doc_scope=has_scope
            )

            # Doc-scoped or deterministic small-talk → shared brain
            if has_scope or not intent.need_retrieval:
                turn = await run_answer_turn(
                    TurnRequest(
                        query=user_text,
                        channel="voice",
                        doc_id=doc_id,
                        category=category,
                        top_k=5,
                        use_llm=True,
                        max_tokens=300,
                        max_context_chars=4000,
                        single_doc_only=doc_id is not None,
                        history=conversation_history,
                    )
                )
                text = turn.answer or self._not_found_response(
                    turn.intent.language
                )
                sources = turn.pack.sources_wire or []
                logger.info(
                    "Nova turn_pipeline: intent=%s persona=%s user=%s... resp=%s...",
                    turn.intent.retrieval_intent,
                    turn.intent.persona,
                    _safe_str(user_text[:50]),
                    _safe_str((text or "")[:80]),
                )
                return VoiceAssistantResult(text=text, sources=sources)

            # Unscoped free-form: Nova pass-1 tool route (legacy fallback)
            if is_small_talk(user_text):
                turn = await run_answer_turn(
                    TurnRequest(
                        query=user_text,
                        channel="voice",
                        use_llm=False,
                        history=conversation_history,
                    )
                )
        if doc_id or category:
            context_text, sources = await self._search_documents(
                query=user_text,
                doc_id=doc_id,
                category=category,
                top_k=5,
            )
            if not context_text.strip():
                return VoiceAssistantResult(
                    text=self._not_found_response(detected_language)
                )
            return await self._generate_contextual_answer(
                user_text=user_text,
                conversation_history=conversation_history,
                context_text=context_text,
                sources=sources,
                detected_language=detected_language,
            )

        # ── Pass 1: Let Nova decide if it needs document search ─────────
        builder_pass1 = NovaPromptBuilder(
            user_query=user_text,
            matches=[],
            conversation_history=conversation_history,
            context_text="",  # No retrieval yet
        )
        system_prompt_pass1 = builder_pass1.system_prompt()

        try:
            client = self._get_ai_client()
            pass1_response = await client.generate_response(
                system_prompt=system_prompt_pass1,
                user_prompt=user_text,
                max_tokens=256,
            )
        except Exception as exc:
            logger.error("LLM pass 1 failed: %s", exc, exc_info=True)
            return VoiceAssistantResult(text=self._error_response(detected_language))

        tool_call = self._parse_tool_call(pass1_response)

        if tool_call is None:
            response = self._clean_for_voice(pass1_response)
            logger.info(
                "Nova direct response: user=%s...  response=%s...",
                _safe_str(user_text[:50]),
                _safe_str(response[:80]),
            )
            return VoiceAssistantResult(text=response)

        search_query = tool_call.get("query", user_text)
        logger.info(
            "Nova requested search_pdf: query=%s...",
            _safe_str(search_query[:80]),
        )

        # Prefer shared pipeline for the search pass
        if getattr(settings, "voice_use_turn_pipeline", True):
            from core.turn_pipeline import TurnRequest, run_answer_turn

            turn = await run_answer_turn(
                TurnRequest(
                    query=search_query or user_text,
                    channel="voice",
                    doc_id=doc_id,
                    category=category,
                    top_k=5,
                    use_llm=True,
                    max_tokens=300,
                    max_context_chars=4000,
                    single_doc_only=doc_id is not None,
                    history=conversation_history,
                )
            )
            return VoiceAssistantResult(
                text=turn.answer or self._not_found_response(detected_language),
                sources=turn.pack.sources_wire or [],
            )

        context_text, sources = await self._search_documents(
            query=search_query,
            doc_id=doc_id,
            category=category,
            top_k=5,
        )

        if not context_text.strip():
            return VoiceAssistantResult(
                text=self._not_found_response(detected_language)
            )

        return await self._generate_contextual_answer(
            user_text=user_text,
            conversation_history=conversation_history,
            context_text=context_text,
            sources=sources,
            detected_language=detected_language,
        )

    async def _generate_contextual_answer(
        self,
        *,
        user_text: str,
        conversation_history: List[Dict[str, str]],
        context_text: str,
        sources: List[Dict[str, Any]],
        detected_language: str,
    ) -> VoiceAssistantResult:
        """Second-pass LLM answer from retrieved document context."""
        builder_pass2 = NovaPromptBuilder(
            user_query=user_text,
            matches=sources,  # for the citation rules the LLM sees
            conversation_history=conversation_history,
            context_text=context_text,
        )
        system_prompt_pass2 = builder_pass2.system_prompt()

        try:
            client = self._get_ai_client()
            pass2_response = await client.generate_response(
                system_prompt=system_prompt_pass2,
                user_prompt=user_text,
                max_tokens=300,
            )
            response = self._clean_for_voice(pass2_response)

            # Safety: if Nova somehow emits another tool call, fall back to
            # the honest "not found" line.
            if self._parse_tool_call(response) is not None:
                response = self._not_found_response(detected_language)

            logger.info(
                "Nova RAG response: user=%s...  response=%s...",
                _safe_str(user_text[:50]),
                _safe_str(response[:80]),
            )
            return VoiceAssistantResult(text=response, sources=sources)

        except Exception as exc:
            logger.error("LLM pass 2 failed: %s", exc, exc_info=True)
            return VoiceAssistantResult(
                text=self._error_response(detected_language),
                sources=sources,
            )

    # ── Retrieval delegation (calls the shared pipeline) ────────────────

    async def _search_documents(
        self,
        query: str,
        *,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Delegate to :func:`core.retrieval.retrieve` and render voice context.

        This is the **only** retrieval entry point for the voice modality.
        It exists to translate the voice engine's kwargs
        (``doc_id``, ``category``) into the retrieval layer's
        :class:`RetrievalScope` dataclass and to render the response in
        the voice-specific citation shape. No retrieval primitives
        (TF-IDF, BM25, RRF, hybrid scoring, etc.) live here — those
        are owned exclusively by :mod:`core.retrieval`.
        """
        try:
            scope = RetrievalScope(
                doc_id=doc_id,
                category=category,
                single_doc_only=doc_id is not None,
            )
            result = await retrieve(query, scope=scope, top_k=top_k)

            if not result.relevant_matches:
                return "", []

            context_text, sources = format_voice_context(
                result.relevant_matches, max_chars=4000
            )
            return context_text, sources

        except Exception as exc:
            logger.error("Document search failed: %s", exc, exc_info=True)
            return "", []

    def _is_small_talk(self, text: str) -> bool:
        """True for turns that should not trigger document retrieval."""
        from core.intent import is_small_talk

        return is_small_talk(text)

    def _not_found_response(self, language: str) -> str:
        """Return a language-appropriate document-not-found response."""
        if language == "ta":
            return "மன்னிக்கவும், பதிவேற்றிய ஆவணங்களில் அந்தத் தகவலைக் கண்டறிய முடியவில்லை."
        if language == "tanglish":
            return "Mannikkavum, upload pannina document-la antha thagaval kandupidikka mudiyala."
        return "I couldn't find that information in the uploaded documents."

    # ── Tool-call parsing ───────────────────────────────────────────────

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect if the LLM response is a ``search_pdf`` tool call.

        Returns the parsed JSON dict if a valid tool call is found,
        else ``None``.
        """
        if not text:
            return None

        stripped = text.strip()

        # Try direct JSON parse first
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and parsed.get("tool") == "search_pdf":
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from ``` blocks or from a bare {...} block
        json_patterns = [
            r"```json\s*(\{[^}]+\})\s*```",
            r"```\s*(\{[^}]+\})\s*```",
            r'(\{"tool"\s*:\s*"search_pdf"[^}]*\})',
        ]
        for pattern in json_patterns:
            match = re.search(pattern, stripped, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict) and parsed.get("tool") == "search_pdf":
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        return None

    # ── Voice-specific post-processing ─────────────────────────────────

    def _clean_for_voice(self, text: str) -> str:
        """Remove formatting / spoken citations (shared implementation)."""
        from core.answer_pack import clean_for_voice

        return clean_for_voice(text)

    def _error_response(self, language: str) -> str:
        """Return a language-appropriate error message."""
        if language == "ta":
            return "மன்னிக்கவும், பதில் உருவாக்கத்தில் பிழை ஏற்பட்டது."
        return "Sorry, I encountered an error generating a response. Please try again."
