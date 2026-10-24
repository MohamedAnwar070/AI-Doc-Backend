"""Document repository — CRUD for the documents/chunks collections."""

from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase


async def save_document(
    db: AsyncIOMotorDatabase,
    job_id: str,
    filename: str,
    content: str,
    embedding: List[float],
    metadata: Optional[dict] = None,
) -> str:
    """Upsert full document metadata and embedding to MongoDB."""
    doc = {
        "job_id": job_id,
        "filename": filename,
        "content": content,
        "normalized_content": " ".join((content or "").split()),
        "embedding": embedding,
        "metadata": metadata or {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.documents.update_one(
        {"job_id": job_id},
        {"$set": doc},
        upsert=True,
    )
    saved = await db.documents.find_one({"job_id": job_id}, {"_id": 1})
    return str(saved["_id"]) if saved else ""


async def get_document_by_job_id(
    db: AsyncIOMotorDatabase, job_id: str
) -> Optional[dict]:
    """Retrieve document by its OCR job ID."""
    return await db.documents.find_one({"job_id": job_id})


async def search_documents(
    db: AsyncIOMotorDatabase, query_filter: dict, limit: int = 10
) -> List[dict]:
    """Basic search in documents collection."""
    cursor = db.documents.find(query_filter).limit(limit)
    return await cursor.to_list(length=limit)


async def save_chunks(
    db: AsyncIOMotorDatabase,
    job_id: str,
    chunks: List[dict],
    *,
    replace_existing: bool = True,
) -> None:
    """Save granular text chunks.

    Chunks are intentionally persisted even when their embedding failed.
    Retrieval can still use them through lexical/BM25/exact matching, and
    operational metadata can reveal degraded embedding health.
    """
    if not chunks:
        if replace_existing:
            await db.chunks.delete_many({"job_id": job_id})
        return

    if replace_existing:
        await db.chunks.delete_many({"job_id": job_id})

    for c in chunks:
        c["job_id"] = job_id
        c["created_at"] = datetime.utcnow()

    await db.chunks.insert_many(chunks)


async def get_all_chunks(
    db: AsyncIOMotorDatabase,
    job_id: Optional[str] = None,
    *,
    with_embeddings_only: bool = False,
) -> List[dict]:
    """Retrieve all searchable chunks, optionally filtered by job_id."""
    query: dict = {}
    if with_embeddings_only:
        query["embedding"] = {"$exists": True, "$not": {"$size": 0}}
    if job_id:
        query["job_id"] = job_id
    cursor = db.chunks.find(query)
    return await cursor.to_list(length=None)


async def get_chunks_by_category(
    db: AsyncIOMotorDatabase,
    category: str,
    job_id: Optional[str] = None,
    *,
    with_embeddings_only: bool = False,
) -> List[dict]:
    """Retrieve chunks filtered by document category (and optionally job_id).

    Used for cross-document search scoped to a specific category (e.g., 'court_orders').
    Only returns chunks that have a stored embedding.
    """
    query: dict = {
        "$or": [{"metadata.category": category}, {"metadata.categories": category}],
    }
    if with_embeddings_only:
        query["embedding"] = {"$exists": True, "$not": {"$size": 0}}
    if job_id:
        query["job_id"] = job_id
    cursor = db.chunks.find(query)
    return await cursor.to_list(length=None)


async def count_chunks(db: AsyncIOMotorDatabase, job_id: str) -> int:
    return int(await db.chunks.count_documents({"job_id": job_id}))


async def count_embedded_chunks(db: AsyncIOMotorDatabase, job_id: str) -> int:
    return int(
        await db.chunks.count_documents(
            {"job_id": job_id, "embedding": {"$exists": True, "$not": {"$size": 0}}}
        )
    )


async def delete_job_records(db: AsyncIOMotorDatabase, job_id: str) -> None:
    await db.documents.delete_one({"job_id": job_id})
    await db.chunks.delete_many({"job_id": job_id})


async def get_chunks_by_ids(
    db: AsyncIOMotorDatabase, job_id: str, chunk_ids: List[int]
) -> List[dict]:
    """Retrieve chunks by job_id and a list of chunk_ids."""
    if not chunk_ids:
        return []
    cursor = db.chunks.find({"job_id": job_id, "chunk_id": {"$in": chunk_ids}})
    return await cursor.to_list(length=None)


async def get_cached_embeddings(
    db: AsyncIOMotorDatabase, hashes: List[str], model_name: str
) -> dict[str, List[float]]:
    """Look up cached embeddings by SHA-256 text hashes for a given model."""
    if not hashes or db is None:
        return {}
    cursor = db.embedding_cache.find(
        {"hash": {"$in": hashes}, "model_name": model_name},
        {"hash": 1, "embedding": 1, "_id": 0},
    )
    docs = await cursor.to_list(length=None)
    return {doc["hash"]: doc["embedding"] for doc in docs if doc.get("embedding")}


async def save_cached_embeddings(
    db: AsyncIOMotorDatabase, records: List[dict]
) -> None:
    """Bulk insert/upsert cached text embeddings into MongoDB."""
    if not records or db is None:
        return
    from pymongo import UpdateOne
    operations = [
        UpdateOne(
            {"hash": r["hash"], "model_name": r["model_name"]},
            {"$set": {**r, "created_at": datetime.utcnow()}},
            upsert=True,
        )
        for r in records
    ]
    try:
        await db.embedding_cache.bulk_write(operations, ordered=False)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Bulk write to embedding_cache failed: %s", exc)


