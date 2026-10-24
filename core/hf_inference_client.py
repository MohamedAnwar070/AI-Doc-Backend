import logging
import base64
from typing import Optional

from config import settings


logger = logging.getLogger(__name__)


def _is_html_response(text: str) -> bool:
    t = (text or "").lstrip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html")


def _httpx_verify() -> bool | str:
    """Single audited entry point for outbound TLS verification.

    Production: ``True`` (system CA bundle).
    Development against self-signed: set ``HTTP_TLS_VERIFY=false``.
    """
    return settings.httpx_verify()


class HFInferenceLlmClient:
    """HuggingFace Inference API client using OpenAI-compatible endpoint"""

    def __init__(self, model: str, api_key: str, base_url: str = "https://router.huggingface.co/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        from openai import AsyncOpenAI
        import httpx

        # We must use AsyncOpenAI and an AsyncClient so we don't block the FastAPI event loop
        http_client = httpx.AsyncClient(verify=_httpx_verify(), timeout=60.0)

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a completion. If system_prompt is given, it is sent as a system message."""
        try:
            logger.info(f"[HF INFERENCE] Calling model={self.model} | base_url={self.base_url}")

            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": prompt})

            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )

            response_text = (resp.choices[0].message.content or "").strip()

            if _is_html_response(response_text):
                raise RuntimeError(
                    "HuggingFace Inference API returned an HTML error page. "
                    "Check HF_INFERENCE_API_KEY is valid and has access."
                )

            return response_text

        except Exception as e:
            msg = str(e)
            if "huggingface.co" in self.base_url and ("403" in msg or "PermissionDenied" in msg or "sufficient permissions" in msg):
                logger.warning(
                    "[HF INFERENCE] Router endpoint denied access (likely 403). "
                    "Falling back to standard HF Inference API."
                )
                return await self._fallback_generate(
                    prompt=prompt, max_tokens=max_tokens, system_prompt=system_prompt
                )

            logger.error(f"[HF INFERENCE ERROR] {e}", exc_info=True)
            raise

    async def get_embedding(self, text: str) -> list[float]:
        """Generate a text embedding using HuggingFace Inference API (router).

        Uses the feature-extraction pipeline endpoint which works for
        sentence-transformers models (e.g. all-MiniLM-L6-v2 → 384-dim).

        Response shapes handled:
          [float, ...]          ← flat 384-element array  (most sentence-transformers)
          [[float, ...]]        ← nested (pooled single sentence)
          [[float, ...], ...]   ← nested (multiple sentences — we take first)
        """
        import httpx

        url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{self.model}/pipeline/feature-extraction"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": text}

        logger.info("[HF EMBEDDING] POST %s", url)

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code >= 400:
            raise RuntimeError(
                f"HF Embedding request failed: {r.status_code} — {r.text[:300]}"
            )

        data = r.json()

        # Shape: flat list of floats  [0.01, -0.03, ...]
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return [float(v) for v in data]

        # Shape: nested list  [[0.01, -0.03, ...]]  or  [[...], [...]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            inner = data[0]
            # Some models return [[token_emb, ...]] — take mean pooling
            if inner and isinstance(inner[0], list):
                # token-level embeddings: average them
                import numpy as np
                return list(np.mean(np.array(inner, dtype=np.float32), axis=0))
            return [float(v) for v in inner]

        logger.warning("[HF EMBEDDING] Unrecognised response shape: %s", str(data)[:200])
        raise RuntimeError(f"Unrecognised HF embedding response shape: {str(data)[:100]}")

    async def _fallback_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Fallback to standard HuggingFace Inference API."""
        import httpx

        url = f"https://router.huggingface.co/hf-inference/models/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_prompt = prompt
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{prompt}"

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.2,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        logger.info(f"[HF INFERENCE FALLBACK] POST {url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code >= 400:
            raise RuntimeError(f"HF Inference API fallback failed: {r.status_code} - {r.text}")

        data = r.json()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first.get("generated_text", "")).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data.get("generated_text", "")).strip()
        return str(data).strip()


class OpenAiLlmClient:
    """
    OpenAI API Client.
    """

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        from openai import AsyncOpenAI
        import httpx

        http_client = httpx.AsyncClient(verify=_httpx_verify(), timeout=60.0)

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a chat completion using direct OpenAI client."""
        try:
            logger.info(f"[OPENAI LLM] Calling model={self.model} | base_url={self.base_url}")

            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": prompt})

            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )

            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            logger.error(f"[OPENAI LLM ERROR] {e}", exc_info=True)
            raise


class BasicAuthLlmClient:
    """
    OpenAI-compatible LLM client using HTTP Basic Authentication.
    """

    def __init__(self, base_url: str, model: str, auth: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._auth_header = ""

        # Encode credentials exactly like curl -u user:pass
        if auth:
            encoded = base64.b64encode(auth.encode()).decode()
            self._auth_header = f"Basic {encoded}"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a chat completion request using direct httpx (like curl -u)."""
        import httpx

        timeout = float(getattr(settings, "ai_model_timeout_seconds", 600.0))
        url = f"{self.base_url}/chat/completions"
        logger.info(f"[BASIC AUTH LLM] POST {url} model={self.model} timeout={timeout}s")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "repetition_penalty": 1.15,
        }

        headers = {"Content-Type": "application/json"}
        if self._auth_header:
            headers["Authorization"] = self._auth_header

        async with httpx.AsyncClient(verify=_httpx_verify(), timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=headers)

        if r.status_code >= 400:
            raise RuntimeError(
                f"BasicAuth LLM request failed: {r.status_code} — {r.text[:300]}"
            )

        data = r.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (content or "").strip()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Batch embeddings via OpenAI-compatible ``/embeddings`` (``input`` list).

        Falls back to one request when the server rejects list input.
        """
        import httpx

        if not texts:
            return []

        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._auth_header:
            headers["Authorization"] = self._auth_header

        payload = {"model": self.model, "input": texts if len(texts) > 1 else texts[0]}
        logger.debug(
            "[BASIC AUTH EMBEDDING] POST %s  model=%s  n=%d",
            url,
            self.model,
            len(texts),
        )

        timeout = float(getattr(settings, "embedding_model_timeout_seconds", 300.0))
        async with httpx.AsyncClient(verify=_httpx_verify(), timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=headers)

        if r.status_code >= 400 and len(texts) > 1:
            # Some servers only accept a single string — fan out.
            logger.warning(
                "Batch embeddings rejected (%s); falling back to sequential",
                r.status_code,
            )
            out: list[list[float]] = []
            for t in texts:
                out.append(await self.get_embedding(t))
            return out

        if r.status_code >= 400:
            raise RuntimeError(
                f"BasicAuth Embedding request failed: {r.status_code} — {r.text[:300]}"
            )

        return self._parse_embeddings_response(r.json(), expected=len(texts))

    async def get_embedding(self, text: str) -> list[float]:
        """Get a single text embedding (wrapper around batch)."""
        results = await self.get_embeddings([text])
        return results[0] if results else []

    @staticmethod
    def _parse_embeddings_response(data, expected: int = 1) -> list[list[float]]:
        """Parse OpenAI / llama-cpp / Ollama embedding response shapes."""
        # Shape 1: OpenAI-standard  {"data": [{"embedding": [...], "index": 0}, ...]}
        if isinstance(data, dict) and "data" in data:
            items = data["data"]
            if isinstance(items, list) and items:
                if isinstance(items[0], dict) and "embedding" in items[0]:
                    # Sort by index when present so batch order is stable.
                    def _idx(it, fallback: int) -> int:
                        try:
                            return int(it.get("index", fallback))
                        except (TypeError, ValueError):
                            return fallback

                    ordered = sorted(
                        enumerate(items),
                        key=lambda pair: _idx(pair[1], pair[0]),
                    )
                    return [list(it.get("embedding") or []) for _, it in ordered]
                if isinstance(items[0], list):
                    return [list(v) for v in items]

        # Shape 2: Flat list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [list(item.get("embedding") or []) for item in data]

        # Shape 3: List of vectors  [[...], [...]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [[float(v) for v in row] for row in data]

        # Shape 4: Single vector as flat list
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return [[float(v) for v in data]]

        # Shape 5: Direct dict  {"embedding": [...]}
        if isinstance(data, dict) and "embedding" in data:
            return [list(data["embedding"])]

        logger.warning(
            "[BASIC AUTH EMBEDDING] Unrecognised response shape: %s",
            str(data)[:200],
        )
        return [[] for _ in range(max(1, expected))]

