"""Context-text formatters for chat and voice modalities.

Both modalities receive the same :class:`core.retrieval.MatchItem`
list from :func:`core.retrieval.retrieve`. They differ only in how
those matches are rendered into the prompt that the LLM sees:

* **Chat** uses ``[Source i] doc_id=X chunk=Y\\n<text>`` — chunk id and
  document id appear so the LLM can produce precise citations when
  asked. Used by :mod:`api.semantic_search`.
* **Voice** uses ``[Page X, Section Y \\u2014 filename]\\n<text>`` —
  page numbers are surfaced directly so the LLM can produce inline
  spoken citations like "(Page 3, Section 2)". Used by
  :mod:`voice.conversation_engine`.

Both formatters are pure string transforms on the data retrieval
returned. They never call the database, the embedding model, or the
LLM. This makes them trivial to unit-test.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from core.retrieval import MatchItem


# ── Voice context (spoken-style with page citations) ───────────────────────

def format_voice_context(
    matches: Iterable[MatchItem],
    *,
    max_chars: int = 4000,
) -> Tuple[str, List[dict]]:
    """Render voice citations in ``[Page X — filename]\\ntext`` style.

    Returns:
        A tuple ``(context_text, sources_list)`` where ``sources_list``
        is a list of plain dicts ready to send to the WebSocket UI.
        Each dict carries the citation metadata the renderer needs:
        score, doc_id, file_path, chunk_id, text, filename, page,
        section_label, source.
    """
    parts: List[str] = []
    sources: List[dict] = []
    used = 0
    for m in matches:
        child_text = (m.text or "").strip()
        if not child_text:
            continue
        context_text = (m.metadata.get("parent_text") or child_text).strip()
        if used >= max_chars:
            break
        remaining = max_chars - used
        if len(context_text) > remaining:
            context_text = context_text[:remaining]

        label = _voice_source_label(m)
        parts.append(f"{label}\n{context_text}" if label else context_text)
        used += len(context_text)

        sources.append(
            {
                "score": float(m.score),
                "doc_id": m.doc_id,
                "file_path": m.file_path
                or f"db://{m.doc_id}/chunk/{m.chunk_id}",
                "chunk_id": m.chunk_id,
                "text": child_text,
                "filename": m.filename,
                "page": m.page,
                "section_label": m.section_label,
                "source": m.source,
            }
        )

    return "\n\n".join(parts), sources


def _voice_source_label(m: MatchItem) -> str:
    """Build the inline citation label the LLM reads before each chunk.

    Examples
    --------
    >>> _voice_source_label(MatchItem(... page=3, filename="doc.pdf", section_label="Rule 5"))
    '[Page 3, Rule 5 — doc.pdf]'
    >>> _voice_source_label(MatchItem(... page=3))
    '[Page 3]'
    >>> _voice_source_label(MatchItem(... filename="doc.pdf"))
    '[doc.pdf]'
    >>> _voice_source_label(MatchItem(...))
    ''
    """
    page = m.page
    section = m.section_label
    filename = m.filename
    if page:
        label = f"[Page {page}"
        if section:
            label += f", {section}"
        if filename:
            label += f" \u2014 {filename}]"
        else:
            label += "]"
        return label
    if filename:
        return f"[{filename}]"
    return ""


# ── Chat context (chunk-id style with source numbering) ────────────────────

def format_chat_context(
    matches: Iterable[MatchItem],
    *,
    max_chars: int = 6000,
    is_local: bool = False,
) -> str:
    """Render chat citations in ``[Source i] doc_id=X chunk=Y\\n<text>`` style.

    The chat ``/api/semantic-search`` endpoint inlines the prompt
    builder's own language directive after this context block, so this
    function returns just the chunk excerpts.

    Args:
        matches: Retrieval matches (post re-ranking).
        max_chars: Hard cap on the rendered text. Excess matches are
            dropped; an over-long first match is truncated to fit.
        is_local: When ``True`` (basic-auth LLM like llama.cpp), each
            chunk's cost is measured in UTF-8 bytes (some local models
            count tokens by bytes). When ``False`` (OpenAI / HF), cost
            is character count.
    """
    blocks: List[str] = []
    used = 0
    for i, m in enumerate(matches, start=1):
        snippet = (m.metadata.get("parent_text") or m.text or "").strip()
        cost = len(snippet.encode("utf-8")) if is_local else len(snippet)
        remaining = max_chars - used
        if remaining <= 0:
            break
        if cost > remaining:
            if used == 0:
                safe_chars = remaining // (3 if is_local else 1)
                snippet = snippet[: max(1, safe_chars)]
            else:
                continue
        blocks.append(f"[Source {i}] doc_id={m.doc_id} chunk={m.chunk_id}\n{snippet}")
        used += cost
    return "\n\n".join(blocks)



__all__ = ["format_chat_context", "format_voice_context"]
