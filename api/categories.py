"""Categories API — list, create, update, delete document categories."""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from pydantic import BaseModel, Field
from typing import Optional

from auth.dependencies import require_admin
from config import settings
from db.mongo import get_db
from db.category_repo import (
    get_all_categories,
    get_category_by_slug,
    get_category_by_id,
    create_category,
    update_category,
    delete_category,
)

router = APIRouter(prefix="/api/categories", tags=["categories"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class CategoryOut(BaseModel):
    id: str
    slug: str
    name: str
    order: int = 0
    is_active: bool
    show_on_homepage: bool = True
    parent_id: Optional[str] = None
    level: int = 1


class CategoryCreate(BaseModel):
    name: str
    order: int = Field(0, ge=0)
    show_on_homepage: bool = True
    parent_id: Optional[str] = None


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    order: Optional[int] = Field(None, ge=0)
    is_active: Optional[bool] = None
    show_on_homepage: Optional[bool] = None
    parent_id: Optional[str] = None


def _normalise_parent_id(parent_id: Optional[str]) -> Optional[str]:
    if parent_id is None:
        return None
    cleaned = parent_id.strip()
    return cleaned or None


async def _resolve_level(db, parent_id: Optional[str]) -> int:
    """Return the level for a category with this parent."""
    if not parent_id:
        return 1

    parent = await get_category_by_id(db, parent_id)
    if not parent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid parent category.",
        )

    parent_level = int(parent.get("level") or 1)
    next_level = parent_level + 1
    if next_level > settings.category_max_depth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Category depth exceeds maximum allowed level "
                f"({settings.category_max_depth})."
            ),
        )
    return next_level


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=List[CategoryOut])
async def list_categories(active_only: bool = True, homepage_only: bool = False):
    """
    Return all document categories.
    Called by both admin upload form and public homepage.
    """
    db = get_db()
    return await get_all_categories(db, active_only=active_only, homepage_only=homepage_only)


@router.post("", response_model=CategoryOut, status_code=status.HTTP_201_CREATED)
async def add_category(payload: CategoryCreate, _admin=Depends(require_admin)):
    """Create a new category."""
    # Auto-generate slug from name: lowercase and spaces to underscores
    generated_slug = payload.name.lower().strip().replace(" ", "_")
    
    db = get_db()
    parent_id = _normalise_parent_id(payload.parent_id)
    # Uniqueness check: same slug under the same parent is not allowed
    dup_query = {"slug": generated_slug, "parent_id": parent_id}
    existing = await db.categories.find_one(dup_query)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A category named '{payload.name}' already exists at this level.",
        )
    
    data = payload.model_dump()
    data["parent_id"] = _normalise_parent_id(data.get("parent_id"))
    data["level"] = await _resolve_level(db, data.get("parent_id"))
    data["slug"] = generated_slug
    return await create_category(db, data)


@router.patch("/{category_id}", response_model=CategoryOut)
async def edit_category(category_id: str, payload: CategoryUpdate, _admin=Depends(require_admin)):
    """Update an existing category's fields."""
    db = get_db()
    current = await get_category_by_id(db, category_id)
    if not current:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found.",
        )

    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update.",
        )

    if "parent_id" in updates:
        parent_id = _normalise_parent_id(updates.get("parent_id"))
        if parent_id == category_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Category cannot be its own parent.",
            )

        child_count = await db.categories.count_documents({"parent_id": category_id})
        if child_count > 0 and parent_id != current.get("parent_id"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot move a category that already has child categories.",
            )

        updates["parent_id"] = parent_id
        updates["level"] = await _resolve_level(db, parent_id)

    updated = await update_category(db, category_id, updates)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found.",
        )
    return updated


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_category(category_id: str, _admin=Depends(require_admin)):
    """Soft-delete a category (marks is_active=False)."""
    db = get_db()
    success = await delete_category(db, category_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found.",
        )
