from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import logging
import json
import os
import re

from core.job_manager import JobManager, get_job_manager
from db.doc_repo import delete_job_records
from db.job_repo import delete_job_meta, get_job_meta
from db.mongo import get_db
from storage.file_handler import FileHandler
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["status"])


class StatusResponse(BaseModel):
    """Response model for status endpoint"""

    job_id: str
    status: str
    pdf_filename: Optional[str] = None
    engine: Optional[str] = None
    total_pages: int
    processed_pages: int
    failed_pages: int
    percentage: float
    time_taken: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    search_indexed: bool = False
    search_index_status: Optional[str] = None
    search_index_started_at: Optional[str] = None
    search_indexed_at: Optional[str] = None
    search_index_error: Optional[str] = None
    search_index_version: Optional[int] = None
    search_chunk_count: int = 0
    search_expected_chunk_count: int = 0
    search_saved_chunk_count: int = 0
    search_embedded_chunk_count: int = 0
    search_embedding_failed_count: int = 0
    search_index_health: Optional[str] = None
    category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sheet_name: Optional[str] = None


class JobSummary(BaseModel):
    """Summary model for listing jobs"""

    job_id: str
    status: str
    pdf_filename: Optional[str] = None
    engine: Optional[str] = None
    total_pages: int
    processed_pages: int
    failed_pages: int
    percentage: float
    time_taken: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    search_indexed: bool = False
    search_index_status: Optional[str] = None
    search_index_started_at: Optional[str] = None
    search_indexed_at: Optional[str] = None
    search_index_error: Optional[str] = None
    search_index_version: Optional[int] = None
    search_chunk_count: int = 0
    search_expected_chunk_count: int = 0
    search_saved_chunk_count: int = 0
    search_embedded_chunk_count: int = 0
    search_embedding_failed_count: int = 0
    search_index_health: Optional[str] = None
    category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sheet_name: Optional[str] = None


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    """
    Get the current status of a job

    Args:
        job_id: Job identifier

    Returns:
        StatusResponse with current job status

    Raises:
        HTTPException: If job not found
    """
    job_manager = get_job_manager()

    try:
        status_info = job_manager.get_status(job_id)

        if status_info.get("status") == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
            )

        return StatusResponse(**job_manager.progress_tracker.get_summary(job_id))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve status: {str(e)}",
        )


class PaginatedJobs(BaseModel):
    items: List[JobSummary]
    total: int
    page: int
    pages: int
    limit: int


class FileSearchResult(JobSummary):
    match_type: str = "filename"
    match_snippet: Optional[str] = None


class FileSearchResponse(BaseModel):
    items: List[FileSearchResult]
    total: int
    query: str


def _job_summary_from_doc(data: dict) -> JobSummary:
    return JobSummary(
        job_id=data.get("job_id"),
        status=data.get("status", "unknown"),
        pdf_filename=data.get("pdf_filename"),
        engine=data.get("engine"),
        total_pages=int(data.get("total_pages", 0)),
        processed_pages=int(data.get("processed_pages", 0)),
        failed_pages=int(data.get("failed_pages", 0)),
        percentage=float(data.get("percentage", 0.0)),
        time_taken=data.get("time_taken"),
        created_at=data.get("created_at"),
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        error=data.get("error"),
        search_indexed=bool(data.get("search_indexed", False)),
        search_index_status=data.get("search_index_status"),
        search_index_started_at=data.get("search_index_started_at"),
        search_indexed_at=data.get("search_indexed_at"),
        search_index_error=data.get("search_index_error"),
        search_index_version=data.get("search_index_version"),
        search_chunk_count=int(data.get("search_chunk_count", 0)),
        search_expected_chunk_count=int(data.get("search_expected_chunk_count", data.get("search_chunk_count", 0))),
        search_saved_chunk_count=int(data.get("search_saved_chunk_count", data.get("search_chunk_count", 0))),
        search_embedded_chunk_count=int(data.get("search_embedded_chunk_count", data.get("search_chunk_count", 0))),
        search_embedding_failed_count=int(data.get("search_embedding_failed_count", 0)),
        search_index_health=data.get("search_index_health"),
        category=data.get("category"),
        categories=data.get("categories") or ([data.get("category")] if data.get("category") else []),
        tags=data.get("tags") or [],
        sheet_name=data.get("sheet_name"),
    )


def _snippet(text: str, query: str, radius: int = 90) -> str:
    if not text:
        return ""
    match = re.search(re.escape(query), text, flags=re.IGNORECASE)
    if not match:
        return text[: radius * 2].strip()
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


@router.get("/jobs/search", response_model=FileSearchResponse)
async def search_jobs(
    q: str,
    limit: int = 50,
    category: Optional[str] = None,
    tag: Optional[str] = None,
) -> FileSearchResponse:
    """
    Plain file-manager search. Matches filenames and stored OCR text without
    embeddings, LLM calls, or semantic retrieval.
    """
    query_text = q.strip()
    if len(query_text) < 2:
        return FileSearchResponse(items=[], total=0, query=query_text)

    db = get_db()
    if db is None:
        return FileSearchResponse(items=[], total=0, query=query_text)

    safe_limit = max(1, min(limit, 100))
    regex = re.escape(query_text)
    category_filter = {}
    if category and category not in {"all", "none"}:
        category_filter["$or"] = [{"category": category}, {"categories": category}]
    elif category == "none":
        category_filter["$and"] = [
            {"category": {"$in": ["", None, "none"]}},
            {"$or": [{"categories": {"$exists": False}}, {"categories": {"$size": 0}}]},
        ]
    if tag and tag != "all":
        category_filter["tags"] = tag

    matches: dict[str, dict] = {}

    filename_filter = {
        **category_filter,
        "pdf_filename": {"$regex": regex, "$options": "i"},
    }
    filename_jobs = await db.jobs.find(filename_filter).sort("created_at", -1).limit(safe_limit).to_list(safe_limit)
    for job in filename_jobs:
        job_id = job.get("job_id")
        if job_id:
            matches[job_id] = {"job": job, "match_type": "filename", "match_snippet": job.get("pdf_filename") or ""}

    tag_search_filter = {k: v for k, v in category_filter.items() if k != "tags"}
    tag_search_filter["tags"] = {"$elemMatch": {"$regex": regex, "$options": "i"}}
    tag_jobs = await db.jobs.find(tag_search_filter).sort("created_at", -1).limit(safe_limit).to_list(safe_limit)
    for job in tag_jobs:
        job_id = job.get("job_id")
        if job_id and job_id not in matches:
            matched_tag = next(
                (t for t in job.get("tags", []) if query_text.lower() in t.lower()), ""
            )
            matches[job_id] = {"job": job, "match_type": "tag", "match_snippet": matched_tag}

    content_job_ids: list[str] = []
    doc_cursor = db.documents.find(
        {"content": {"$regex": regex, "$options": "i"}},
        {"job_id": 1, "content": 1},
    ).limit(safe_limit)
    for doc in await doc_cursor.to_list(safe_limit):
        job_id = doc.get("job_id")
        if job_id and job_id not in matches:
            content_job_ids.append(job_id)
            matches[job_id] = {
                "job": None,
                "match_type": "content",
                "match_snippet": _snippet(doc.get("content", ""), query_text),
            }

    if len(matches) < safe_limit:
        remaining = safe_limit - len(matches)
        chunk_cursor = db.chunks.find(
            {"text": {"$regex": regex, "$options": "i"}},
            {"job_id": 1, "text": 1},
        ).limit(remaining * 3)
        for chunk in await chunk_cursor.to_list(remaining * 3):
            job_id = chunk.get("job_id")
            if not job_id or job_id in matches:
                continue
            content_job_ids.append(job_id)
            matches[job_id] = {
                "job": None,
                "match_type": "content",
                "match_snippet": _snippet(chunk.get("text", ""), query_text),
            }
            if len(matches) >= safe_limit:
                break

    if content_job_ids:
        job_filter = {"job_id": {"$in": content_job_ids}, **category_filter}
        jobs_by_id = {
            job.get("job_id"): job
            for job in await db.jobs.find(job_filter).to_list(len(content_job_ids))
        }
        for job_id in list(matches.keys()):
            if matches[job_id]["job"] is None:
                job = jobs_by_id.get(job_id)
                if job:
                    matches[job_id]["job"] = job
                else:
                    matches.pop(job_id, None)

    items: list[FileSearchResult] = []
    for match in matches.values():
        try:
            summary = _job_summary_from_doc(match["job"])
            items.append(FileSearchResult(
                **summary.dict(),
                match_type=match["match_type"],
                match_snippet=match["match_snippet"],
            ))
        except Exception:
            continue

    return FileSearchResponse(items=items[:safe_limit], total=len(items), query=query_text)


@router.get("/jobs", response_model=PaginatedJobs)
async def list_jobs(
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    category: Optional[str] = None,
    tag: Optional[str] = None
) -> PaginatedJobs:
    """
    List jobs with pagination using MongoDB indexing.
    """
    from db.mongo import get_db
    from db.job_repo import list_jobs_paginated

    db = get_db()
    result = await list_jobs_paginated(
        db,
        page=page,
        limit=limit,
        status=status,
        category=category,
        tag=tag
    )
    
    # Map raw items to JobSummary objects
    summaries = []
    for data in result["items"]:
        try:
            summaries.append(_job_summary_from_doc(data))
        except Exception:
            continue
            
    return PaginatedJobs(
        items=summaries,
        total=result["total"],
        page=result["page"],
        pages=result["pages"],
        limit=result["limit"]
    )


@router.get("/jobs/stats")
async def get_jobs_stats() -> dict:
    """
    Return aggregate counts: total jobs and count per category.
    Used by FileManager to show item counts on folder icons.
    """
    from db.mongo import get_db
    db = get_db()
    if db is None:
        return {"total": 0, "by_category": {}}
    try:
        total = await db.jobs.count_documents({})
        pipeline = [
            {
                "$project": {
                    "safe_categories": {
                        "$cond": [
                            {"$isArray": "$categories"},
                            "$categories",
                            [],
                        ]
                    },
                    "category": 1,
                }
            },
            {
                "$project": {
                    "category_values": {
                        "$cond": [
                            {"$gt": [{"$size": "$safe_categories"}, 0]},
                            "$safe_categories",
                            [{"$ifNull": ["$category", ""]}],
                        ]
                    }
                }
            },
            {"$unwind": "$category_values"},
            {"$group": {"_id": "$category_values", "count": {"$sum": 1}}},
        ]
        raw = await db.jobs.aggregate(pipeline).to_list(None)
        by_category: dict = {}
        for item in raw:
            key = item.get("_id") or ""
            by_category[key] = item.get("count", 0)
        return {"total": total, "by_category": by_category}
    except Exception as e:
        return {"total": 0, "by_category": {}, "error": str(e)}



@router.post("/resume/{job_id}")
async def resume_job(job_id: str) -> dict:
    """
    Resume a stopped or failed job

    Args:
        job_id: Job identifier

    Returns:
        Success message

    Raises:
        HTTPException: If job not found or cannot be resumed
    """
    job_manager = get_job_manager()

    # Check if job exists and can be resumed
    if not job_manager.get_result(job_id) and not job_manager.get_status(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )

    status_info = job_manager.get_status(job_id)
    if status_info.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )

    try:
        job_manager.resolve_source_path(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    success = job_manager.resume_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} cannot be resumed (it might be completed or invalid)",
        )

    return {"message": f"Job {job_id} resumed", "job_id": job_id, "status": "running"}


@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job

    Args:
        job_id: Job identifier

    Returns:
        Success message

    Raises:
        HTTPException: If job not found
    """
    job_manager = get_job_manager()

    success = job_manager.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )

    return {"message": f"Job {job_id} has been cancelled", "job_id": job_id}


@router.delete("/delete/{job_id}")
async def delete_job(job_id: str) -> dict:
    """
    Delete a job and all associated files

    Args:
        job_id: Job identifier

    Returns:
        Success message

    Raises:
        HTTPException: If job not found
    """
    job_manager = get_job_manager()
    db = get_db()

    # Job may still exist in Mongo listing even if local output dir was removed.
    job_meta = await get_job_meta(db, job_id) if db is not None else None

    success = job_manager.delete_job(job_id)

    if not success and not job_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )

    if db is not None:
        await delete_job_records(db, job_id)
        await delete_job_meta(db, job_id)

    return {
        "message": f"Job {job_id} and all associated files have been deleted",
        "job_id": job_id,
    }


@router.post("/search-index/reindex/{job_id}")
async def reindex_job(job_id: str, force: bool = True) -> dict:
    """Rebuild search chunks + embeddings for one job (can take minutes on BGE-M3)."""
    job_manager = get_job_manager()
    try:
        result = await job_manager.index_job(job_id, force=force)
        return {"message": f"Indexed job {job_id}", **result}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to reindex {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index job: {e}",
        )


# Full-corpus re-embed (MiniLM → BGE-M3) can take far longer than nginx allows.
# Default: schedule in background and return immediately (avoids 504).
_backfill_state: dict = {
    "running": False,
    "force": None,
    "started_at": None,
    "finished_at": None,
    "last_result": None,
    "last_error": None,
}


async def _run_backfill_task(force: bool) -> None:
    import time

    _backfill_state["running"] = True
    _backfill_state["force"] = force
    _backfill_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _backfill_state["finished_at"] = None
    _backfill_state["last_error"] = None
    job_manager = get_job_manager()
    try:
        logger.info("search-index backfill started (force=%s)", force)
        result = await job_manager.backfill_search_indexes(force=force)
        _backfill_state["last_result"] = result
        logger.info(
            "search-index backfill finished: indexed=%s skipped=%s failed=%s",
            result.get("indexed"),
            result.get("skipped"),
            result.get("failed"),
        )
    except Exception as exc:
        logger.error("search-index backfill failed: %s", exc, exc_info=True)
        _backfill_state["last_error"] = str(exc)
        _backfill_state["last_result"] = None
    finally:
        _backfill_state["running"] = False
        _backfill_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


@router.post("/search-index/backfill")
async def backfill_search_indexes(
    force: bool = False,
    background: bool = True,
) -> dict:
    """Re-index all completed jobs (chunk + embed via current embedding backend).

    * ``force=true`` — rebuild even if already indexed (required after model change).
    * ``background=true`` (default) — return immediately; poll status endpoint.
    * ``background=false`` — wait until done (may 504 behind nginx).
    """
    import asyncio
    import time

    if background:
        if _backfill_state.get("running"):
            return {
                "message": "Backfill already running",
                "status": "running",
                **{k: v for k, v in _backfill_state.items() if k != "last_result"},
            }
        asyncio.create_task(_run_backfill_task(force))
        return {
            "message": "Backfill scheduled in background",
            "status": "scheduled",
            "force": force,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "poll": "GET /api/search-index/backfill/status",
        }

    job_manager = get_job_manager()
    try:
        result = await job_manager.backfill_search_indexes(force=force)
        return {"message": "Backfill completed", "status": "completed", **result}
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backfill failed: {e}",
        )


@router.get("/search-index/backfill/status")
async def backfill_search_indexes_status() -> dict:
    """Poll long-running backfill (after POST .../backfill?background=true)."""
    return {
        "status": "running" if _backfill_state.get("running") else "idle",
        "force": _backfill_state.get("force"),
        "started_at": _backfill_state.get("started_at"),
        "finished_at": _backfill_state.get("finished_at"),
        "last_error": _backfill_state.get("last_error"),
        "last_result": _backfill_state.get("last_result"),
    }
