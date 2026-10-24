"""Tag repository — CRUD for the tags collection."""
from typing import List
from datetime import datetime
import re
from pymongo.errors import DuplicateKeyError
from motor.motor_asyncio import AsyncIOMotorDatabase


def _normalise_tag_doc(tag: dict) -> dict:
    tag["id"] = str(tag.pop("_id"))
    return tag


async def get_all_tags(db: AsyncIOMotorDatabase) -> List[dict]:
    """Return all tags, sorted alphabetically."""
    cursor = db.tags.find({}).sort("name", 1)
    tags = await cursor.to_list(length=None)
    return [_normalise_tag_doc(tag) for tag in tags]


async def get_or_create_tags(db: AsyncIOMotorDatabase, names: List[str]) -> List[str]:
    """Ensure each tag name exists (case-insensitively) in the tags collection.

    Returns the cleaned tag names, reusing the stored casing of any tag that
    already exists so "Invoice" and "invoice" don't create two tags.
    """
    cleaned: List[str] = []
    for raw in names:
        name = (raw or "").strip()
        if name and name.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(name)

    resolved: List[str] = []
    for name in cleaned:
        existing = await db.tags.find_one(
            {"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}
        )
        if existing:
            resolved.append(existing["name"])
        else:
            try:
                await db.tags.insert_one({"name": name, "created_at": datetime.utcnow()})
                resolved.append(name)
            except DuplicateKeyError:
                # A concurrent upload may create the same tag between our
                # lookup and insert. Resolve it instead of failing the upload.
                existing = await db.tags.find_one({"name": name})
                resolved.append(existing["name"] if existing else name)
    return resolved
