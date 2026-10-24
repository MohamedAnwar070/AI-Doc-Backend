"""Category repository — CRUD for the categories collection."""
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

# ── Default seed categories ───────────────────────────────────────────────────

DEFAULT_CATEGORIES = [
    {
        "slug": "act's",
        "name": "Act's",
        "order": 10,
        "show_on_homepage": True,
        "parent_id": None,
        "level": 1,
    },
    {
        "slug": "audit",
        "name": "Audit",
        "order": 20,
        "show_on_homepage": True,
        "parent_id": None,
        "level": 1,
    },
]


def _normalise_category_doc(cat: dict) -> dict:
    """Convert Mongo internals to API-friendly fields with safe defaults."""
    parent_val = cat.get("parent_id")
    if isinstance(parent_val, ObjectId):
        parent_val = str(parent_val)
    cat["parent_id"] = parent_val
    cat["level"] = int(cat.get("level") or 1)
    cat["order"] = int(cat.get("order") or 0)
    cat["show_on_homepage"] = bool(cat.get("show_on_homepage", True))
    cat["id"] = str(cat.pop("_id"))
    return cat


async def seed_default_categories(db: AsyncIOMotorDatabase) -> None:
    """Insert default categories if the collection is empty."""
    count = await db.categories.count_documents({})
    if count == 0:
        now = datetime.utcnow()
        docs = [
            {**cat, "is_active": True, "created_at": now}
            for cat in DEFAULT_CATEGORIES
        ]
        await db.categories.insert_many(docs)


async def get_all_categories(
    db: AsyncIOMotorDatabase, active_only: bool = True, homepage_only: bool = False
) -> List[dict]:
    """Return all categories, sorted by order."""
    query = {}
    if active_only:
        query["is_active"] = True
    if homepage_only:
        query["show_on_homepage"] = {"$ne": False}
        
    cursor = db.categories.find(query).sort([
        ("level", 1),
        ("order", 1),
        ("name", 1),
        ("created_at", -1),
    ])
    categories = await cursor.to_list(length=None)
    for cat in categories:
        _normalise_category_doc(cat)
    return categories


async def get_category_by_slug(
    db: AsyncIOMotorDatabase, slug: str
) -> Optional[dict]:
    """Find a single category by its slug."""
    cat = await db.categories.find_one({"slug": slug})
    if cat:
        _normalise_category_doc(cat)
    return cat


async def get_category_by_id(
    db: AsyncIOMotorDatabase, category_id: str
) -> Optional[dict]:
    """Find a single category by its MongoDB ObjectId string."""
    if not ObjectId.is_valid(category_id):
        return None
    cat = await db.categories.find_one({"_id": ObjectId(category_id)})
    if cat:
        _normalise_category_doc(cat)
    return cat


async def create_category(db: AsyncIOMotorDatabase, data: dict) -> dict:
    """Insert a new category. Returns the created document."""
    data["created_at"] = datetime.utcnow()
    data.setdefault("is_active", True)
    data.setdefault("show_on_homepage", True)
    data.setdefault("parent_id", None)
    data.setdefault("level", 1)
    data.setdefault("order", 0)
    result = await db.categories.insert_one(data)
    created = await db.categories.find_one({"_id": result.inserted_id})
    _normalise_category_doc(created)
    return created


async def update_category(
    db: AsyncIOMotorDatabase, category_id: str, updates: dict
) -> Optional[dict]:
    """Update a category by its MongoDB ObjectId string."""
    if not ObjectId.is_valid(category_id):
        return None
    await db.categories.update_one(
        {"_id": ObjectId(category_id)}, {"$set": updates}
    )
    updated = await db.categories.find_one({"_id": ObjectId(category_id)})
    if updated:
        _normalise_category_doc(updated)
    return updated


async def delete_category(db: AsyncIOMotorDatabase, category_id: str) -> bool:
    """Soft-delete a category (set is_active=False)."""
    result = await db.categories.update_one(
        {"_id": ObjectId(category_id)}, {"$set": {"is_active": False}}
    )
    return result.modified_count > 0
