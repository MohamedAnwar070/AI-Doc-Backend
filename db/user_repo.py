"""User repository — CRUD for the users collection."""
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from passlib.context import CryptContext
import bcrypt

# Monkeypatch to fix passlib/bcrypt 4.0.0+ compatibility issue
# where passlib expects bcrypt.__about__.__version__
if not hasattr(bcrypt, "__about__"):
    bcrypt.__about__ = type("About", (object,), {"__version__": bcrypt.__version__})

from typing import Optional
from db.models import UserInDB, UserCreate, UserPublic
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash(password: str) -> str:
    # bcrypt has a 72-byte limit for passwords.
    return pwd_context.hash(password[:72])


def _to_public(doc: dict) -> UserPublic:
    return UserPublic(
        id=str(doc["_id"]),
        username=doc["username"],
        email=doc["email"],
        role=doc.get("role", "user"),
        is_active=doc.get("is_active", True),
        created_at=doc.get("created_at", datetime.utcnow()),
    )


async def create_user(db: AsyncIOMotorDatabase, data: UserCreate) -> UserPublic:
    existing = await db.users.find_one({"email": data.email})
    if existing:
        raise ValueError("Email already registered")
    doc = {
        "username": data.username,
        "email": data.email,
        "hashed_password": _hash(data.password),
        "role": data.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
    }
    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _to_public(doc)


async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[dict]:
    return await db.users.find_one({"email": email})


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[UserPublic]:
    try:
        doc = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    if not doc:
        return None
    return _to_public(doc)


def verify_password(plain: str, hashed: str) -> bool:
    # bcrypt has a 72-byte limit for passwords.
    return pwd_context.verify(plain[:72], hashed)
