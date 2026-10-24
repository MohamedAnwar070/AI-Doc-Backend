"""Upload API — `POST /api/upload`.

Production-quality upload endpoint with the following hardening:

* **Filename sanitisation**: rejects path traversal attempts (``..``,
  absolute paths, slashes, backslashes), strips control characters,
  and stores using the original sanitized filename.
  See :func:`api._upload_utils.sanitise_filename`.
* **Content sniff**: confirms the upload starts with ``%PDF-`` rather
  than trusting the client-supplied extension only.
* **Size enforcement**: streamed in 8 KiB chunks with a hard byte cap
  honoured *during* the copy, so an attacker cannot bypass the
  ``Content-Length``-based check by lying about the header.

The on-disk filename is always under ``settings.upload_dir``, never
above it. The original (sanitised) filename is preserved in MongoDB
metadata and in the OCR job record so downstream UIs still display
the right name.
"""
from __future__ import annotations

import logging
import zipfile
import os
import uuid
import re
from pathlib import Path

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from api._upload_utils import (
    COPY_CHUNK_BYTES,
    PDF_MAGIC,
    XLSX_MAGIC,
    sanitise_filename,
    validate_language,
)
from config import settings
from core.job_manager import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])
_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _validate_upload_id(upload_id: str) -> str:
    value = (upload_id or "").strip()
    if value and not _UPLOAD_ID_RE.fullmatch(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload_id",
        )
    return value


def _normalise_categories(category: str = "", categories: str | list[str] | None = None) -> tuple[str, list[str]]:
    """Return primary category plus full selected category list."""
    values: list[str] = []

    def add(raw: str | None) -> None:
        if not raw:
            return
        text = raw.strip()
        if not text or text == "none":
            return
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    for item in parsed:
                        add(str(item))
                    return
            except Exception:
                pass
        for part in text.split(","):
            cleaned = part.strip()
            if cleaned and cleaned != "none" and cleaned not in values:
                values.append(cleaned)

    add(category)
    if isinstance(categories, str):
        add(categories)
    else:
        for item in categories or []:
            add(item)

    return (values[0] if values else ""), values


def _normalise_tags(tags: str | list[str] | None = None) -> list[str]:
    """Parse a tags form field (JSON array or comma-separated text) into a clean list."""
    values: list[str] = []

    def add(raw: str | None) -> None:
        if not raw:
            return
        text = raw.strip()
        if not text:
            return
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    for item in parsed:
                        add(str(item))
                    return
            except Exception:
                pass
        for part in text.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)

    if isinstance(tags, str):
        add(tags)
    else:
        for item in tags or []:
            add(item)

    return values


class UploadResponse(BaseModel):
    """Response body for ``POST /api/upload``."""

    job_id: str
    filename: str
    total_pages: int
    status: str
    message: str


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _stream_to_disk(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
) -> int:
    """Copy ``upload`` to ``destination`` while enforcing ``max_bytes``.

    Returns the number of bytes written. Raises ``HTTPException(413)``
    if the limit is exceeded mid-copy.
    """
    written = 0
    with destination.open("wb") as buffer:
        while True:
            block = await upload.read(COPY_CHUNK_BYTES)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                buffer.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File size exceeds maximum allowed size of "
                        f"{settings.max_upload_size_mb}MB"
                    ),
                )
            buffer.write(block)
    return written


def _is_pdf(path: Path) -> bool:
    """Return True iff ``path`` starts with the PDF magic bytes."""
    try:
        with path.open("rb") as fh:
            head = fh.read(len(PDF_MAGIC))
    except OSError:
        return False
    return head.startswith(PDF_MAGIC)


def _is_xlsx(path: Path) -> bool:
    """Return True iff ``path`` starts with the ZIP magic bytes (xlsx is a zip)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(len(XLSX_MAGIC))
    except OSError:
        return False
    return head.startswith(XLSX_MAGIC)


def _is_docx(path: Path) -> bool:
    """Return True iff ``path`` has the required DOCX package entries."""
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return "[Content_Types].xml" in names and "word/document.xml" in names


# ── Endpoint ────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF, DOCX, or XLSX file to process"),
    lang: str = Form("auto", description="OCR language (auto, en, ta, ...)"),
    doc_type: str = Form("auto", description="Document type (auto, native, scanned)"),
    category: str = Form("", description="Document category slug (e.g. court_orders)"),
    categories: Optional[str] = Form(None, description="Document category slugs as JSON or comma-separated text"),
    tags: Optional[str] = Form(None, description="File tags as JSON array or comma-separated text"),
    upload_id: str = Form("", description="Stable client upload ID for idempotent retries"),
) -> UploadResponse:
    """Accept a PDF or DOCX upload and start a processing job for it."""
    upload_id = _validate_upload_id(upload_id)
    fname = (file.filename or "").lower()
    if not (fname.endswith(".pdf") or fname.endswith(".xlsx") or fname.endswith(".docx")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, DOCX, and XLSX files are allowed",
        )

    job_manager = get_job_manager()
    existing = job_manager.find_job_by_client_upload_id(upload_id)
    if existing:
        if existing.get("status") in {"pending", "running"}:
            job_manager.enqueue_job(existing["job_id"])
        await file.close()
        return UploadResponse(
            job_id=existing["job_id"],
            filename=existing.get("pdf_filename", file.filename or ""),
            total_pages=existing.get("total_pages", 0),
            status=existing.get("status", "pending"),
            message="Upload already accepted; returning the existing job.",
        )

    safe_name = sanitise_filename(file.filename)
    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    transfer_id = uuid.uuid4().hex
    partial_path = (upload_dir / f".{transfer_id}.part").resolve()
    upload_path = (upload_dir / f"{transfer_id}-{safe_name}").resolve()

    # Defensive: confirm the resolved path is still inside upload_dir.
    if upload_dir not in upload_path.parents and upload_path != upload_dir:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )

    max_bytes = settings.get_max_upload_size_bytes()

    try:
        bytes_written = await _stream_to_disk(file, partial_path, max_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover — disk / IO faults
        logger.error("Upload failed while writing %s: %s", partial_path, exc)
        partial_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed",
        ) from exc
    finally:
        await file.close()

    is_pdf = fname.endswith(".pdf") and _is_pdf(partial_path)
    is_xlsx = fname.endswith(".xlsx") and _is_xlsx(partial_path)
    is_docx = fname.endswith(".docx") and _is_docx(partial_path)

    if not (is_pdf or is_xlsx or is_docx):
        partial_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid PDF, DOCX, or XLSX",
        )

    os.replace(partial_path, upload_path)
    logger.info(
        "Uploaded file: %s (%d bytes) -> %s",
        file.filename,
        bytes_written,
        upload_path.name,
    )

    primary_category, category_list = _normalise_categories(category, categories)
    tag_list = _normalise_tags(tags)
    if tag_list:
        from db.mongo import get_db
        from db.tag_repo import get_or_create_tags

        tag_list = await get_or_create_tags(get_db(), tag_list)
    try:
        if is_docx:
            job_id = job_manager.create_word_job(
                docx_path=upload_path,
                docx_filename=safe_name,
                category=primary_category,
                categories=category_list,
                tags=tag_list,
                client_upload_id=upload_id,
            )
            job_manager.start_word_processing(job_id, upload_path)
        elif is_xlsx:
            from core.excel_worker import list_sheets

            sheets = list_sheets(upload_path)
            if not sheets:
                raise ValueError("No sheets found in XLSX")
            sheet_name = sheets[0]
            job_id = job_manager.create_excel_job(
                xlsx_path=upload_path,
                xlsx_filename=safe_name,
                sheet_name=sheet_name,
                category=primary_category,
                categories=category_list,
                tags=tag_list,
                client_upload_id=upload_id,
            )
            job_manager.start_excel_processing(job_id, upload_path, sheet_name)
        else:
            job_id = job_manager.create_job(
                pdf_path=upload_path,
                pdf_filename=safe_name,
                ocr_lang=validate_language(lang),
                doc_type=doc_type,
                category=primary_category,
                categories=category_list,
                tags=tag_list,
                client_upload_id=upload_id,
            )
            job_manager.start_processing(job_id, upload_path)
        status_info = job_manager.get_status(job_id)
    except ValueError as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise

    response = UploadResponse(
        job_id=job_id,
        filename=safe_name,
        total_pages=status_info.get("total_pages", 0),
        status=status_info.get("status", "pending"),
        message=(
            f"File uploaded successfully. Processing started with "
            f"{'word' if is_docx else 'excel' if is_xlsx else settings.ocr_engine} engine."
        ),
    )
    upload_path.unlink(missing_ok=True)
    return response


# ── Excel-specific endpoints ─────────────────────────────────────────────────

import tempfile as _tempfile


@router.post("/excel/sheets", tags=["upload"])
async def get_excel_sheets(
    file: UploadFile = File(..., description="XLSX file to inspect"),
) -> dict:
    """Return the list of sheet names in an uploaded XLSX without creating a job.

    The client uploads the file once to this endpoint to discover sheet names,
    then calls ``POST /api/upload/excel`` with the chosen sheet_name.
    """
    if not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only XLSX files are accepted")

    # Write to a temp file so openpyxl can open it
    tmp = _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    try:
        written = 0
        while True:
            block = await file.read(COPY_CHUNK_BYTES)
            if not block:
                break
            written += len(block)
            if written > 50 * 1024 * 1024:  # 50 MB safety limit
                raise HTTPException(status_code=413, detail="File too large for sheet inspection")
            tmp.write(block)
        tmp.flush()
        tmp.close()

        from core.excel_worker import list_sheets
        sheets = list_sheets(Path(tmp.name))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read XLSX: {exc}") from exc
    finally:
        await file.close()
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    return {"sheets": sheets}


@router.post("/upload/excel", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED, tags=["upload"])
async def upload_excel(
    file: UploadFile = File(..., description="XLSX file to process"),
    sheet_name: str = Form(..., description="Sheet name to extract"),
    category: str = Form("", description="Document category slug"),
    categories: Optional[str] = Form(None, description="Document category slugs as JSON or comma-separated text"),
    tags: Optional[str] = Form(None, description="File tags as JSON array or comma-separated text"),
    upload_id: str = Form("", description="Stable client upload ID for idempotent retries"),
) -> UploadResponse:
    """Upload an XLSX file and start an ExcelWorker job for the given sheet."""
    upload_id = _validate_upload_id(upload_id)
    if not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only XLSX files are allowed")

    job_manager = get_job_manager()
    existing = job_manager.find_job_by_client_upload_id(upload_id)
    if existing:
        if existing.get("status") in {"pending", "running"}:
            job_manager.enqueue_job(existing["job_id"])
        await file.close()
        return UploadResponse(
            job_id=existing["job_id"],
            filename=existing.get("pdf_filename", file.filename or ""),
            total_pages=existing.get("total_pages", 1),
            status=existing.get("status", "pending"),
            message="Upload already accepted; returning the existing job.",
        )

    safe_name = sanitise_filename(file.filename)
    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    transfer_id = uuid.uuid4().hex
    partial_path = (upload_dir / f".{transfer_id}.part").resolve()
    upload_path = (upload_dir / f"{transfer_id}-{safe_name}").resolve()

    if upload_dir not in upload_path.parents and upload_path != upload_dir:
        raise HTTPException(status_code=400, detail="Invalid filename")

    max_bytes = settings.get_max_upload_size_bytes()
    try:
        bytes_written = await _stream_to_disk(file, partial_path, max_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Excel upload failed while writing %s: %s", partial_path, exc)
        partial_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Upload failed") from exc
    finally:
        await file.close()

    if not _is_xlsx(partial_path):
        partial_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid XLSX")

    os.replace(partial_path, upload_path)

    logger.info("Excel uploaded: %s (%d bytes) sheet=%r", safe_name, bytes_written, sheet_name)

    primary_category, category_list = _normalise_categories(category, categories)
    tag_list = _normalise_tags(tags)
    if tag_list:
        from db.mongo import get_db
        from db.tag_repo import get_or_create_tags

        tag_list = await get_or_create_tags(get_db(), tag_list)
    try:
        job_id = job_manager.create_excel_job(
            xlsx_path=upload_path,
            xlsx_filename=safe_name,
            sheet_name=sheet_name,
            category=primary_category,
            categories=category_list,
            tags=tag_list,
            client_upload_id=upload_id,
        )
        job_manager.start_excel_processing(job_id, upload_path, sheet_name)
        status_info = job_manager.get_status(job_id)
    except ValueError as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise

    response = UploadResponse(
        job_id=job_id,
        filename=safe_name,
        total_pages=status_info.get("total_pages", 1),
        status=status_info.get("status", "pending"),
        message=f"Excel sheet '{sheet_name}' upload queued for processing.",
    )
    upload_path.unlink(missing_ok=True)
    return response
