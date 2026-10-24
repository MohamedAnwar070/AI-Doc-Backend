"""Chat repository — CRUD for saved_chats collection."""
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.models import SavedChat, ChatMessage
from typing import List, Optional
from datetime import datetime


def _doc_to_model(doc: dict) -> SavedChat:
    doc["_id"] = str(doc["_id"])
    return SavedChat(**doc)


async def create_chat(db: AsyncIOMotorDatabase, user_id: str, job_id: str, title: str) -> SavedChat:
    doc = {
        "user_id": user_id,
        "job_id": job_id,
        "title": title,
        "messages": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = await db.saved_chats.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _doc_to_model(doc)


async def get_user_chats(db: AsyncIOMotorDatabase, user_id: str, job_id: Optional[str] = None) -> List[SavedChat]:
    query = {"user_id": user_id}
    if job_id:
        query["job_id"] = job_id
    cursor = db.saved_chats.find(query).sort("updated_at", -1)
    return [_doc_to_model(doc) async for doc in cursor]


async def get_chat(db: AsyncIOMotorDatabase, chat_id: str, user_id: str) -> Optional[SavedChat]:
    try:
        doc = await db.saved_chats.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    except Exception:
        return None
    return _doc_to_model(doc) if doc else None


async def append_message(
    db: AsyncIOMotorDatabase, chat_id: str, user_id: str, message: ChatMessage
) -> Optional[SavedChat]:
    try:
        result = await db.saved_chats.find_one_and_update(
            {"_id": ObjectId(chat_id), "user_id": user_id},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": datetime.utcnow()},
            },
            return_document=True,
        )
    except Exception:
        return None
    return _doc_to_model(result) if result else None


async def delete_chat(db: AsyncIOMotorDatabase, chat_id: str, user_id: str) -> bool:
    try:
        result = await db.saved_chats.delete_one({"_id": ObjectId(chat_id), "user_id": user_id})
        return result.deleted_count == 1
    except Exception:
        return False
