"""Unified LLM client factory for chat and voice.

This is the **single source of truth** for which model answers a user
question. Both the chat ``/api/semantic-search`` endpoint and the
``voice.conversation_engine`` go through :func:`get_ai_client` and
share the exact same provider selection logic.

Why centralise
--------------
The previous setup had two factories:

* ``core.llm.get_ai_client()`` — used by voice.
* Inline ``if switch_ai == "ai_model": ...`` ladder inside
  ``api/semantic_search.py`` — used by chat.

They drifted: chat's ladder had a different fallback chain (it would
prefer ``ai_model`` over HF even when the basic-auth URL was empty,
because the chat endpoint checked ``api_key and model`` instead of
``url and auth``). This caused voice and chat to silently answer
through different model deployments.

The factory below picks exactly one provider per request based on
``SWITCH_AI``:

==================== ============================================
``SWITCH_AI`` value    Resolves to
==================== ============================================
``open_ai``           :class:`OpenAiLlmClient` using OpenAI key
``ai_model``          :class:`BasicAuthLlmClient` using the active
                      ``AI_MODEL_RUNTIME`` endpoint
``hf_inference``      :class:`HFInferenceLlmClient` using the HF key
(any) / fallback      HF Inference when no other config is set
==================== ============================================

Both paths construct an :class:`AIClient` wrapper that exposes a
single ``async generate_response(system_prompt, user_prompt,
max_tokens) -> str`` method. Callers never need to know which provider
is in play.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings
from core.hf_inference_client import (
    BasicAuthLlmClient,
    HFInferenceLlmClient,
    OpenAiLlmClient,
)

logger = logging.getLogger(__name__)


class AIClient:
    """Thin async wrapper around the configured LLM provider.

    Exposes ``generate_response(system_prompt, user_prompt, max_tokens)``
    so callers don't have to know whether they're hitting OpenAI,
    llama.cpp via Basic Auth, or HuggingFace Inference. The underlying
    client is selected once at construction by :func:`get_ai_client`.
    """

    def __init__(self, client, provider_label: str):
        self._client = client
        self.provider_label = provider_label

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Call the underlying provider with a uniform signature."""
        return await self._client.generate(
            user_prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )


def get_ai_client(preferred: Optional[str] = None) -> AIClient:
    """Return a ready-to-call :class:`AIClient` for the configured model.

    Args:
        preferred: Optional explicit override for ``settings.switch_ai``.
            Used by tests and by callers that want to bypass the env
            config (e.g. an admin endpoint that wants to force OpenAI
            even when ``SWITCH_AI=ai_model``). When ``None``, reads
            ``settings.switch_ai``.

    Returns:
        An :class:`AIClient` bound to whichever provider is configured.

    Raises:
        RuntimeError: If no provider can be resolved from the current
            settings. The error message tells the operator which env
            vars to set.
    """
    switch_ai = (preferred or getattr(settings, "switch_ai", "") or "").strip().lower()

    # ── 1. OpenAI ────────────────────────────────────────────────────────
    if switch_ai == "open_ai":
        client = _try_openai()
        if client is not None:
            return client

    # ── 2. Local "AI model" (llama.cpp / vLLM with Basic Auth) ─────────
    if switch_ai == "ai_model":
        client = _try_ai_model()
        if client is not None:
            return client

    # ── 3. HF Inference (fallback for ``switch_ai=hf_inference`` or empty) ─
    client = _try_hf_inference()
    if client is not None:
        return client

    # ── 4. Final fallback — if SWITCH_AI was explicit but config was
    # missing, surface a precise error. If SWITCH_AI was empty, try the
    # other providers in priority order before giving up.
    if switch_ai in ("open_ai", "ai_model", "hf_inference"):
        raise RuntimeError(
            f"SWITCH_AI={switch_ai!r} is set but no valid credentials are "
            f"configured for it. Check the relevant env vars."
        )

    # SWITCH_AI was empty — try every provider in order before giving up.
    for builder in (_try_openai, _try_ai_model):
        client = builder()
        if client is not None:
            return client

    raise RuntimeError(
        "No AI client configured. Set one of:\n"
        "  - SWITCH_AI=open_ai     with OPENAI_API_KEY / LLM_MODEL_OPENAI / LLM_API_BASE_OPENAI\n"
        "  - SWITCH_AI=ai_model    with AI_MODEL_RUNTIME plus AI_MODEL_LOCAL_* or AI_MODEL_REMOTE_*\n"
        "  - SWITCH_AI=hf_inference with HF_INFERENCE_API_KEY / HF_INFERENCE_MODEL"
    )


# ── Provider constructors ───────────────────────────────────────────────────

def _try_openai() -> Optional[AIClient]:
    api_key = (getattr(settings, "openai_api_key", "") or "").strip()
    if not api_key:
        logger.debug("OpenAI provider not configured (no api_key)")
        return None
    base = (
        getattr(settings, "llm_api_base_openai", "https://api.openai.com/v1")
        or "https://api.openai.com/v1"
    ).strip()
    model = (
        getattr(settings, "llm_model_openai", "gpt-3.5-turbo") or "gpt-3.5-turbo"
    ).strip()
    if model.startswith("openai/"):
        model = model.replace("openai/", "", 1)
    logger.info("LLM provider: openai (model=%s)", model)
    return AIClient(
        OpenAiLlmClient(model=model, api_key=api_key, base_url=base),
        provider_label="openai",
    )


def _try_ai_model() -> Optional[AIClient]:
    base_url = settings.active_ai_model_url()
    auth = settings.active_ai_model_auth()
    model = settings.active_ai_model_model()
    if not base_url:
        logger.debug(
            "AI-model provider not configured (runtime=%s, url=%r)",
            getattr(settings, "ai_model_runtime", "remote"),
            bool(base_url),
        )
        return None
    logger.info(
        "LLM provider: ai_model (runtime=%s, url=%s, model=%s)",
        getattr(settings, "ai_model_runtime", "remote"),
        base_url,
        model,
    )
    return AIClient(
        BasicAuthLlmClient(base_url=base_url, model=model, auth=auth),
        provider_label="ai_model",
    )


def _try_hf_inference() -> Optional[AIClient]:
    api_key = (getattr(settings, "hf_inference_api_key", "") or "").strip()
    model = (getattr(settings, "hf_inference_model", "") or "").strip()
    base_url = (getattr(settings, "hf_inference_base_url", "") or "").strip()
    if not api_key or not model:
        logger.debug(
            "HF Inference provider not configured (key=%r, model=%r)",
            bool(api_key),
            bool(model),
        )
        return None
    logger.info("LLM provider: hf_inference (model=%s)", model)
    return AIClient(
        HFInferenceLlmClient(model=model, api_key=api_key, base_url=base_url),
        provider_label="hf_inference",
    )
