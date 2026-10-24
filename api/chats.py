"""Saved chats API — /api/chats/*."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from db.mongo import get_db
from db.models import UserPublic, SavedChat, ChatMessage
from db import chat_repo
from auth.dependencies import get_current_user

router = APIRouter(prefix="/api/chats", tags=["chats"])


class CreateChatRequest(BaseModel):
    job_id: str
    title: str


class AppendMessageRequest(BaseModel):
    role: str   # "user" | "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@router.post("", response_model=SavedChat, status_code=201)
async def create_chat(
    body: CreateChatRequest,
    current_user: UserPublic = Depends(get_current_user),
):
    db = get_db()
    return await chat_repo.create_chat(db, current_user.id, body.job_id, body.title)


@router.get("", response_model=List[SavedChat])
async def list_chats(job_id: Optional[str] = None, current_user: UserPublic = Depends(get_current_user)):
    db = get_db()
    return await chat_repo.get_user_chats(db, current_user.id, job_id)


@router.get("/{chat_id}", response_model=SavedChat)
async def get_chat(chat_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_db()
    chat = await chat_repo.get_chat(db, chat_id, current_user.id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/{chat_id}/messages", response_model=SavedChat)
async def append_message(
    chat_id: str,
    body: AppendMessageRequest,
    current_user: UserPublic = Depends(get_current_user),
):
    db = get_db()
    msg = ChatMessage(role=body.role, content=body.content, metadata=body.metadata)
    chat = await chat_repo.append_message(db, chat_id, current_user.id, msg)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_db()
    ok = await chat_repo.delete_chat(db, chat_id, current_user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
