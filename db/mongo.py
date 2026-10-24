"""MongoDB motor client — async, singleton."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import settings
import logging

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_uri)
        logger.info("MongoDB client created: %s", settings.mongodb_uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[settings.mongodb_db_name]


# Alias for backwards-compat
db_client = get_client


async def connect_db():
    """Initialise MongoDB connection and ensure all required indexes exist."""
    db = get_db()

    # ── documents collection ────────────────────────────────────────────────
    await db.documents.create_index("job_id", unique=True)
    await db.documents.create_index(
        [("filename", "text"), ("content", "text")],
        name="document_text_search",
        default_language="none",
    )

    # ── chunks collection ───────────────────────────────────────────────────
    await db.chunks.create_index([("job_id", 1), ("chunk_id", 1)], unique=True)
    await db.chunks.create_index("job_id")
    await db.chunks.create_index(
        [("filename", "text"), ("text", "text"), ("metadata.section_label", "text")],
        name="chunk_text_search",
        default_language="none",
    )
    # ── jobs collection ───────────────────────────────────────────────────
    await db.jobs.create_index("job_id", unique=True)
    await db.jobs.create_index("created_at")
    await db.jobs.create_index("category")
    await db.jobs.create_index("categories")
    await db.jobs.create_index("tags")
    await db.jobs.create_index("status")

    # ── tags collection ─────────────────────────────────────────────────────
    await db.tags.create_index("name", unique=True)

    # ── embedding_cache collection ──────────────────────────────────────────
    await db.embedding_cache.create_index(
        [("hash", 1), ("model_name", 1)],
        unique=True,
        name="embedding_cache_hash_model",
    )

    logger.info("MongoDB indexes ensured.")


async def close_db():
    """Close MongoDB connection."""
    global _client
    if _client:
        _client.close()
        _client = None
