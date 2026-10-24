"""Page-level translation API for processed OCR text."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import google.auth
import httpx
from fastapi import APIRouter, HTTPException, status
from google.auth.transport.requests import Request as GoogleAuthRequest
from pydantic import BaseModel

from config import settings
from core.job_manager import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["translation"])

TargetLanguage = Literal["en", "ta"]


class TranslatePageRequest(BaseModel):
    target_language: TargetLanguage


class TranslatePageResponse(BaseModel):
    job_id: str
    page: int
    source_language: Optional[str] = None
    target_language: TargetLanguage
    translated_text: str
    source_hash: str
    cached: bool
    updated_at: str


class TranslateBatchRequest(BaseModel):
    target_language: TargetLanguage
    pages: list[int]


class TranslateBatchResponse(BaseModel):
    job_id: str
    target_language: TargetLanguage
    items: list[TranslatePageResponse]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _translation_dir(job_id: str) -> Path:
    path = Path(settings.output_dir) / job_id / "translations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _translation_cache_path(job_id: str, page: int, target_language: str) -> Path:
    return _translation_dir(job_id) / f"page_{page:04d}_{target_language}.json"


def _plain_text_from_html(value: str) -> str:
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(p|div|tr|h[1-6]|li)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _block_text(block: dict) -> str:
    if not isinstance(block, dict):
        return ""
    text = str(block.get("text") or "").strip()
    if text:
        return text
    html_text = block.get("html")
    if html_text:
        return _plain_text_from_html(str(html_text))
    return ""


def _page_text_from_structure(page_data: dict) -> str:
    blocks = page_data.get("blocks")
    if not isinstance(blocks, list):
        blocks = page_data.get("elements")
    if not isinstance(blocks, list):
        return ""
    return "\n\n".join(text for text in (_block_text(block) for block in blocks) if text)


def _page_text_from_flat_result(job_id: str, page: int) -> str:
    result_path = Path(settings.output_dir) / job_id / "result.txt"
    if not result_path.exists():
        return ""
    text = result_path.read_text(encoding="utf-8", errors="ignore")
    marker_pattern = re.compile(r"^---\s*Page\s+(\d+)\s*---\s*$", re.IGNORECASE | re.MULTILINE)
    matches = list(marker_pattern.finditer(text))
    if not matches:
        return text.strip() if page == 1 else ""

    for index, match in enumerate(matches):
        if int(match.group(1)) != page:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[start:end].strip()
    return ""


def _get_current_page_text(job_id: str, page: int) -> str:
    job_manager = get_job_manager()
    page_data = job_manager.storage.load_page_structure(job_id, page)
    if page_data:
        page_text = _page_text_from_structure(page_data)
        if page_text:
            return page_text
    return _page_text_from_flat_result(job_id, page)


def _source_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _detect_simple_page_language(text: str) -> str:
    """Classify page text enough to avoid same-language translation calls.

    Returns ``en``, ``ta``, ``mixed`` or ``unknown``. This is deliberately
    conservative: mixed English/Tamil pages still go through translation.
    """
    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    signal_chars = tamil_chars + latin_chars

    if signal_chars == 0:
        return "unknown"

    tamil_ratio = tamil_chars / signal_chars
    latin_ratio = latin_chars / signal_chars

    if tamil_ratio >= 0.9:
        return "ta"
    if latin_ratio >= 0.9:
        return "en"
    return "mixed"


def _should_skip_translation(source_language: str, target_language: str) -> bool:
    return source_language in {"en", "ta"} and source_language == target_language


def _chunk_text(text: str, max_chars: int = 4500) -> list[str]:
    paragraphs = re.split(r"(\n{2,})", text)
    chunks: list[str] = []
    current = ""

    for part in paragraphs:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for index in range(0, len(part), max_chars):
                chunks.append(part[index:index + max_chars])
            continue

        if current and len(current) + len(part) > max_chars:
            chunks.append(current)
            current = part
        else:
            current += part

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def _get_google_translate_token() -> tuple[str, str]:
    try:
        credentials_path = (
            settings.google_translate_credentials
            or settings.google_docai_credentials
            or settings.google_visionai_credentials
        )
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        credentials, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-translation"]
        )
        credentials.refresh(GoogleAuthRequest())

        project_id = settings.google_translate_project_id or detected_project or settings.google_docai_project_id
        if not project_id:
            raise RuntimeError("Google Cloud project ID is not configured")

        return credentials.token, project_id
    except Exception as exc:
        logger.error("Failed to authenticate with Google Translation API: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Translation is not configured. Set GOOGLE_TRANSLATE_PROJECT_ID and Google credentials.",
        ) from exc


async def _translate_with_google_cloud(text: str, target_language: str) -> tuple[str, Optional[str]]:
    token, project_id = _get_google_translate_token()
    location = settings.google_translate_location or "global"
    url = (
        f"https://translation.googleapis.com/v3/projects/{project_id}"
        f"/locations/{location}:translateText"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    chunks = _chunk_text(text)
    if not chunks:
        return "", None

    payload = {
        "contents": chunks,
        "mimeType": "text/plain",
        "targetLanguageCode": target_language,
    }

    async with httpx.AsyncClient(timeout=settings.google_translate_timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            logger.error("Google Translation API returned an error: %s", detail)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Google Translation API failed.",
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Google Translation API request failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not reach Google Translation API.",
            ) from exc

    data = response.json()
    translations = data.get("translations") or []
    translated_chunks = [item.get("translatedText", "") for item in translations]
    source_language = translations[0].get("detectedLanguageCode") if translations else None
    return "\n\n".join(translated_chunks), source_language


async def _translate_with_free_google(text: str, target_language: str) -> tuple[str, Optional[str]]:
    """Translate via Google's public web translate endpoint.

    This avoids service-account setup and keeps page translations persisted in
    our own cache. It is intentionally isolated behind the provider switch so
    production deployments can move to Cloud Translation without touching UI.
    """
    translated_chunks: list[str] = []
    detected_language: Optional[str] = None

    async with httpx.AsyncClient(timeout=settings.google_translate_timeout) as client:
        for chunk in _chunk_text(text, max_chars=3500):
            try:
                response = await client.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={
                        "client": "gtx",
                        "sl": "auto",
                        "tl": target_language,
                        "dt": "t",
                        "q": chunk,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.error("Free Google translate request failed: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Free Google translation failed.",
                ) from exc

            data = response.json()
            translated = "".join(part[0] for part in (data[0] or []) if part and part[0])
            translated_chunks.append(translated)
            if not detected_language and len(data) > 2 and isinstance(data[2], str):
                detected_language = data[2]

    return "\n\n".join(translated_chunks), detected_language


async def _translate_text(text: str, target_language: str) -> tuple[str, Optional[str]]:
    provider = settings.google_translate_provider
    if provider == "free":
        return await _translate_with_free_google(text, target_language)
    if provider == "cloud":
        return await _translate_with_google_cloud(text, target_language)

    try:
        return await _translate_with_google_cloud(text, target_language)
    except HTTPException as exc:
        logger.warning("Cloud translation unavailable, falling back to free endpoint: %s", exc.detail)
        return await _translate_with_free_google(text, target_language)


def _response_from_cache(job_id: str, page: int, target_language: TargetLanguage, current_hash: str, cached: dict) -> TranslatePageResponse:
    return TranslatePageResponse(
        job_id=job_id,
        page=page,
        source_language=cached.get("source_language"),
        target_language=target_language,
        translated_text=cached.get("translated_text", ""),
        source_hash=current_hash,
        cached=True,
        updated_at=cached.get("updated_at") or cached.get("created_at") or _utc_now(),
    )


def _write_translation_cache(
    job_id: str,
    page: int,
    target_language: TargetLanguage,
    source_hash: str,
    translated_text: str,
    source_language: Optional[str],
    translation_skipped: bool,
) -> TranslatePageResponse:
    now = _utc_now()
    cache_path = _translation_cache_path(job_id, page, target_language)
    cache_payload = {
        "job_id": job_id,
        "page": page,
        "source_language": source_language,
        "target_language": target_language,
        "source_hash": source_hash,
        "translated_text": translated_text,
        "translation_skipped": translation_skipped,
        "created_at": now,
        "updated_at": now,
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return TranslatePageResponse(
        job_id=job_id,
        page=page,
        source_language=source_language,
        target_language=target_language,
        translated_text=translated_text,
        source_hash=source_hash,
        cached=False,
        updated_at=now,
    )


async def _translate_one_page(job_id: str, page: int, target_language: TargetLanguage) -> TranslatePageResponse:
    if page < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Page must be >= 1")

    page_text = _get_current_page_text(job_id, page)
    if not page_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No processed text found for page {page}.",
        )

    current_hash = _source_hash(page_text)
    cache_path = _translation_cache_path(job_id, page, target_language)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("source_hash") == current_hash:
                return _response_from_cache(job_id, page, target_language, current_hash, cached)
        except Exception as exc:
            logger.warning("Ignoring invalid translation cache %s: %s", cache_path, exc)

    detected_page_language = _detect_simple_page_language(page_text)
    skip_translation = _should_skip_translation(detected_page_language, target_language)
    if skip_translation:
        translated_text = page_text
        source_language = detected_page_language
    else:
        translated_text, source_language = await _translate_text(page_text, target_language)
        source_language = source_language or detected_page_language

    return _write_translation_cache(
        job_id=job_id,
        page=page,
        target_language=target_language,
        source_hash=current_hash,
        translated_text=translated_text,
        source_language=source_language,
        translation_skipped=skip_translation,
    )


async def _ensure_job_exists(job_id: str) -> None:
    job_manager = get_job_manager()
    status_info = job_manager.get_status(job_id)
    if status_info.get("status") == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")


@router.post(
    "/result/{job_id}/translate/page/{page}",
    response_model=TranslatePageResponse,
    status_code=status.HTTP_200_OK,
)
async def translate_result_page(
    job_id: str,
    page: int,
    body: TranslatePageRequest,
) -> TranslatePageResponse:
    """Translate one processed OCR page and persist a source-hash-aware cache."""
    await _ensure_job_exists(job_id)
    return await _translate_one_page(job_id, page, body.target_language)


@router.post(
    "/result/{job_id}/translate/batch",
    response_model=TranslateBatchResponse,
    status_code=status.HTTP_200_OK,
)
async def translate_result_pages_batch(
    job_id: str,
    body: TranslateBatchRequest,
) -> TranslateBatchResponse:
    """Translate a small batch of processed pages while preserving per-page cache."""
    await _ensure_job_exists(job_id)
    seen: set[int] = set()
    pages: list[int] = []
    for page in body.pages:
        if page >= 1 and page not in seen:
            seen.add(page)
            pages.append(page)

    if not pages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one valid page is required")
    if len(pages) > 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch can include at most 8 pages")

    items: list[TranslatePageResponse] = []
    for page in pages:
        try:
            items.append(await _translate_one_page(job_id, page, body.target_language))
        except HTTPException as exc:
            if exc.status_code == status.HTTP_404_NOT_FOUND:
                logger.warning("Skipping untranslated missing page %s for job %s: %s", page, job_id, exc.detail)
                continue
            raise

    return TranslateBatchResponse(
        job_id=job_id,
        target_language=body.target_language,
        items=items,
    )
