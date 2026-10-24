"""Tags API — list all known tags for autocomplete/filtering."""
from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

from db.mongo import get_db
from db.tag_repo import get_all_tags

router = APIRouter(prefix="/api/tags", tags=["tags"])


class TagOut(BaseModel):
    id: str
    name: str


@router.get("", response_model=List[TagOut])
async def list_tags():
    """
    Return all tags. Used by the upload form for autocomplete/inline-create
    and by the file manager for filtering the file list.
    """
    db = get_db()
    return await get_all_tags(db)
