"""Pydantic models for MongoDB documents."""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


# ── User ────────────────────────────────────────────────────────────────────

class UserInDB(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    username: str
    email: str
    hashed_password: str
    role: Literal["admin", "user"] = "user"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: Literal["admin", "user"] = "user"


class UserPublic(BaseModel):
    id: str
    username: str
    email: str
    role: Literal["admin", "user"] = "user"
    is_active: bool
    created_at: datetime


# ── Saved Chat ───────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str           # "user" | "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SavedChat(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    job_id: str
    title: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


# ── Document Embedding ────────────────────────────────────────────────────────

class EmbeddingChunk(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    job_id: str
    doc_id: str
    chunk_id: int
    text: str
    embedding: List[float]
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


# ── Document Category ─────────────────────────────────────────────────────────

class DocumentCategory(BaseModel):
    """A document category stored in MongoDB (categories collection)."""
    id: Optional[str] = Field(None, alias="_id")
    slug: str        # machine-readable key, e.g. "court_orders"
    name: str        # human-readable label, e.g. "Court Orders"
    description: Optional[str] = None
    order: int = 0   # display order in UI
    is_active: bool = True
    show_on_homepage: bool = True
    parent_id: Optional[str] = None
    level: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


class DocumentCategoryCreate(BaseModel):
    slug: str
    name: str
    description: Optional[str] = None
    order: int = 0
    show_on_homepage: bool = True
    parent_id: Optional[str] = None


class DocumentCategoryPublic(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    order: int
    is_active: bool
    show_on_homepage: bool = True
    parent_id: Optional[str] = None
    level: int = 1
    created_at: datetime


# ── Tag ─────────────────────────────────────────────────────────────────────

class Tag(BaseModel):
    """A file tag stored in MongoDB (tags collection)."""
    id: Optional[str] = Field(None, alias="_id")
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


class TagPublic(BaseModel):
    id: str
    name: str
    created_at: datetime
