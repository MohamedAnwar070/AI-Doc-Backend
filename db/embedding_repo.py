"""Embedding repository — store and retrieve vector chunks in MongoDB."""
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.models import EmbeddingChunk
from typing import List, Optional
from datetime import datetime
import numpy as np


async def upsert_chunks(db: AsyncIOMotorDatabase, chunks: List[EmbeddingChunk]) -> int:
    """Insert or replace all chunks for a job_id+doc_id. Returns inserted count."""
    if not chunks:
        return 0
    # Delete old chunks for same job_id
    job_id = chunks[0].job_id
    await db.chunks.delete_many({"job_id": job_id})
    docs = []
    for c in chunks:
        d = c.model_dump(exclude={"id"})
        docs.append(d)
    result = await db.chunks.insert_many(docs)
    return len(result.inserted_ids)


async def get_chunks_for_job(db: AsyncIOMotorDatabase, job_id: str) -> List[EmbeddingChunk]:
    cursor = db.chunks.find({"job_id": job_id})
    out = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        out.append(EmbeddingChunk(**doc))
    return out


async def similarity_search(
    db: AsyncIOMotorDatabase,
    job_id: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[dict]:
    """Brute-force cosine similarity over stored chunks for a job."""
    chunks = await get_chunks_for_job(db, job_id)
    if not chunks:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []

    scored = []
    for chunk in chunks:
        v = np.array(chunk.embedding, dtype=np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        score = float(np.dot(q, v) / (q_norm * v_norm))
        scored.append({"score": score, "chunk": chunk})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [
        {"score": s["score"], "text": s["chunk"].text, "metadata": s["chunk"].metadata}
        for s in scored[:top_k]
    ]
