"""LLM-based OCR text cleaner.

Sends OCR-extracted text to the configured LLM (``AI_MODEL_URL``) for
post-processing. The model is instructed to **only fix OCR noise** —
garbled characters, broken words, mis-recognised glyphs — without
altering meaning, structure, or removing any content.

This module is **synchronous** because it runs inside the
``ProcessPoolExecutor`` child processes used by the OCR worker.  It
uses ``httpx`` directly (not the async ``BasicAuthLlmClient``) to
avoid pulling an event loop into the worker subprocess.

Configuration
-------------
Reads from ``config.Settings``:

*  ``llm_clean_enabled`` — master switch.
*  ``llm_clean_max_chars`` — chunk size limit per request.
*  active ``AI_MODEL_RUNTIME`` endpoint from config — LLM endpoint.
"""
from __future__ import annotations

import base64
import logging
import textwrap
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_TIMEOUT_SECONDS: float = 90.0

_SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are an OCR post-processor. Your ONLY job is to fix OCR errors in the text below.

    Rules you MUST follow:
    1. Fix garbled characters, broken words, and mis-recognised letters.
    2. Fix spacing issues (merged words, extra spaces within words).
    3. Preserve the EXACT meaning — never add, remove, or rephrase content.
    4. Preserve paragraph breaks, bullet points, numbering, and structure.
    5. Preserve all bilingual content (Tamil, English, or any other language) as-is.
    6. Do NOT add any commentary, explanation, or markdown formatting.
    7. Do NOT wrap your output in quotes or code blocks.
    8. Return ONLY the corrected text, nothing else.
""")

_USER_PROMPT_TEMPLATE: str = "Fix OCR errors in this text:\n\n{text}"


# ── Public API ───────────────────────────────────────────────────────────────

def is_llm_clean_enabled() -> bool:
    """Check if LLM cleaning is enabled (reads settings at call time)."""
    from config import settings
    if not settings.llm_clean_enabled:
        return False

    switch_ai = settings.switch_ai
    if switch_ai == "open_ai":
        return bool(
            settings.openai_api_key
            and settings.llm_api_base_openai
            and settings.llm_model_openai
        )
    elif switch_ai == "huggingface":
        return bool(
            settings.hf_inference_api_key
            and settings.hf_inference_model
        )
    else:  # settings.switch_ai == "ai_model"
        return bool(
            settings.active_ai_model_url()
            and settings.active_ai_model_model()
        )


def llm_clean_text(text: str) -> str:
    """Clean OCR text through the LLM. Returns original on failure.

    If the text is longer than ``llm_clean_max_chars`` it is split into
    chunks, each cleaned independently, then reassembled.
    """
    if not text or not text.strip():
        return text

    if not is_llm_clean_enabled():
        return text

    from config import settings
    max_chars = settings.llm_clean_max_chars

    try:
        if len(text) <= max_chars:
            return _clean_chunk(text)

        # Split into chunks on paragraph boundaries
        chunks = _split_into_chunks(text, max_chars)
        cleaned_parts: List[str] = []
        for i, chunk in enumerate(chunks):
            logger.debug("LLM-clean chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            cleaned_parts.append(_clean_chunk(chunk))
        return "\n".join(cleaned_parts)

    except Exception as exc:
        logger.warning("LLM clean failed, returning original text: %s", exc)
        return text


def llm_clean_elements(elements: List[dict]) -> List[dict]:
    """Clean the ``text`` field of each structured element in-place.

    Skips table elements (HTML is not suitable for LLM cleaning) and
    empty text fields.
    """
    if not is_llm_clean_enabled():
        return elements

    for el in elements:
        el_type = el.get("type", "")
        el_text = el.get("text", "")
        if el_type == "table" or not el_text or not el_text.strip():
            continue
        el["text"] = llm_clean_text(el_text)
    return elements


# ── Internal helpers ─────────────────────────────────────────────────────────

def _clean_chunk(text: str) -> str:
    """Send a single chunk to the LLM and return the cleaned text."""
    import httpx
    from config import settings

    switch_ai = settings.switch_ai

    if switch_ai == "open_ai":
        url = f"{settings.llm_api_base_openai.rstrip('/')}/chat/completions"
        model = settings.llm_model_openai.strip()
        if model.startswith("openai/"):
            model = model.replace("openai/", "", 1)
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
    elif switch_ai == "huggingface":
        url = f"{settings.hf_inference_base_url.rstrip('/')}/chat/completions"
        model = settings.hf_inference_model.strip()
        headers = {
            "Authorization": f"Bearer {settings.hf_inference_api_key}",
            "Content-Type": "application/json",
        }
    else:  # settings.switch_ai == "ai_model"
        url = f"{settings.active_ai_model_url().rstrip('/')}/chat/completions"
        model = settings.active_ai_model_model()
        active_auth = settings.active_ai_model_auth()
        headers = {
            "Content-Type": "application/json",
        }
        if active_auth:
            encoded_auth = base64.b64encode(active_auth.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        "max_tokens": max(len(text) * 2, 1024),
        "temperature": 0.1,
    }

    verify: bool | str = True
    try:
        verify = settings.httpx_verify()
    except Exception:
        pass

    with httpx.Client(timeout=_TIMEOUT_SECONDS, verify=verify) as client:
        response = client.post(url, json=payload, headers=headers)

    if response.status_code >= 400:
        raise RuntimeError(
            f"LLM clean request failed: HTTP {response.status_code} — "
            f"{response.text[:300]}"
        )

    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    cleaned = (content or "").strip()

    # Sanity: if LLM returned empty or drastically shorter, keep original
    if not cleaned or len(cleaned) < len(text) * 0.3:
        logger.warning(
            "LLM clean returned suspiciously short output (%d vs %d chars); "
            "keeping original",
            len(cleaned),
            len(text),
        )
        return text

    return cleaned


def _split_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks ≤ max_chars, preferring paragraph boundaries."""
    paragraphs = text.split("\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 1  # +1 for the newline
        if current_len + para_len > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n".join(current))

    return chunks
