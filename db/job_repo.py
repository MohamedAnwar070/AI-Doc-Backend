"""Job repository — metadata mirroring for pagination & efficient listing."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

async def upsert_job_meta(
    db: AsyncIOMotorDatabase,
    job_id: str,
    data: Dict[str, Any]
) -> None:
    """Synchronize job metadata to MongoDB jobs collection."""
    # Build filter to match job
    filt = {"job_id": job_id}
    
    # We only want to store metadata needed for listing/filtering
    # Extract clean summary fields
    doc = {
        "job_id": job_id,
        "pdf_filename": data.get("pdf_filename"),
        "status": data.get("status", "unknown"),
        "total_pages": int(data.get("total_pages", 0)),
        "processed_pages": int(data.get("processed_pages_count", 0)),
        "failed_pages": int(data.get("failed_pages_count", 0)),
        "percentage": float(data.get("percentage", 0.0)),
        "engine": data.get("engine"),
        "category": data.get("category", ""),
        "categories": data.get("categories") or ([data.get("category")] if data.get("category") else []),
        "tags": data.get("tags") or [],
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "completed_at": data.get("completed_at"),
        "error": data.get("error"),
        "search_indexed": bool(data.get("search_indexed", False)),
        "search_index_status": data.get("search_index_status"),
        "search_index_started_at": data.get("search_index_started_at"),
        "search_indexed_at": data.get("search_indexed_at"),
        "search_index_error": data.get("search_index_error"),
        "search_index_version": data.get("search_index_version"),
        "search_chunk_count": int(data.get("search_chunk_count", 0)),
        "search_expected_chunk_count": int(data.get("search_expected_chunk_count", data.get("search_chunk_count", 0))),
        "search_saved_chunk_count": int(data.get("search_saved_chunk_count", data.get("search_chunk_count", 0))),
        "search_embedded_chunk_count": int(data.get("search_embedded_chunk_count", data.get("search_chunk_count", 0))),
        "search_embedding_failed_count": int(data.get("search_embedding_failed_count", 0)),
        "search_index_health": data.get("search_index_health"),
        "time_taken": data.get("time_taken"),
        "sheet_name": data.get("sheet_name"),
        "last_updated": datetime.utcnow().isoformat()
    }
    
    # Ensure float for percentage
    try:
        doc["percentage"] = float(doc["percentage"])
    except:
        doc["percentage"] = 0.0

    await db.jobs.update_one(filt, {"$set": doc}, upsert=True)

async def list_jobs_paginated(
    db: AsyncIOMotorDatabase,
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    sort_by: str = "created_at",
    sort_dir: int = -1
) -> Dict[str, Any]:
    """Retrieve a paginated list of jobs from MongoDB."""
    query = {}
    if status and status != "all":
        query["status"] = status
    if category and category != "all" and category != "none":
        query["$or"] = [{"category": category}, {"categories": category}]
    elif category == "none":
        query["$and"] = [
            {"category": {"$in": ["", None, "none"]}},
            {"$or": [{"categories": {"$exists": False}}, {"categories": {"$size": 0}}]},
        ]
    if tag and tag != "all":
        query["tags"] = tag

    skip = (page - 1) * limit
    
    # Get total count for pagination metadata
    total = await db.jobs.count_documents(query)
    
    # Fetch items
    cursor = db.jobs.find(query).sort(sort_by, sort_dir).skip(skip).limit(limit)
    items = await cursor.to_list(length=limit)
    
    # Format IDs
    for item in items:
        item["id"] = str(item.pop("_id"))
        
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit if limit > 0 else 0
    }

async def get_job_meta(db: AsyncIOMotorDatabase, job_id: str) -> Optional[dict]:
    """Fetch metadata for a single job."""
    return await db.jobs.find_one({"job_id": job_id})

async def delete_job_meta(db: AsyncIOMotorDatabase, job_id: str) -> bool:
    """Remove job metadata from index."""
    res = await db.jobs.delete_one({"job_id": job_id})
    return res.deleted_count > 0
