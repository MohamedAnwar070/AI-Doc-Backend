"""Auth API — `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`.

Hardening notes (production-quality):

* The public ``/register`` endpoint **never** lets a caller self-promote
  to ``role='admin'`` unless ``settings.allow_public_admin_registration``
  is true (development only). This is the textbook "vertical privilege
  escalation" defence.
* All admin accounts must be created either by another admin (via
  ``POST /api/admin/users``) or by the bootstrap script
  ``scripts/create_admin.py`` — *not* by anonymous HTTP.
* On login, we always return generic 401s ("Incorrect email or
  password") so we don't leak which emails exist.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.jwt import create_access_token
from config import settings
from db.models import UserCreate, UserPublic
from db.mongo import get_db
from db.user_repo import _to_public, create_user, get_user_by_email, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Login request body."""

    email: str
    password: str


class TokenResponse(BaseModel):
    """Login success response containing the JWT and the user record."""

    access_token: str
    token_type: str = "bearer"
    role: str
    user: UserPublic


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
)
async def register(data: UserCreate) -> UserPublic:
    """Create a new end-user account.

    Anonymous callers can only ever create ``role='user'`` accounts.
    Requests asking for ``role='admin'`` from the public endpoint are
    silently downgraded to ``user`` (in production) or accepted only
    when the explicit dev flag is enabled.
    """
    sanitised = data.model_copy(update={
        "role": _resolve_public_role(data.role),
    })

    db = get_db()
    try:
        user = await create_user(db, sanitised)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return user


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest) -> TokenResponse:
    """Authenticate by email + password and return a signed JWT."""
    db = get_db()
    doc = await get_user_by_email(db, data.email)
    if not doc or not verify_password(data.password, doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if doc.get("is_active", True) is False:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token({"sub": str(doc["_id"])})
    user = _to_public(doc)
    return TokenResponse(access_token=token, role=user.role, user=user)


@router.get("/me", response_model=UserPublic)
async def me(
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    """Return the currently-authenticated user's public profile."""
    return current_user


# ── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_public_role(requested: str) -> str:
    """Decide which role anonymous registration is allowed to create.

    Always returns ``'user'`` in production. Only returns ``'admin'``
    when the operator has explicitly opted-in via configuration —
    typically only on local dev or first-boot bootstrap.
    """
    if requested == "admin" and not settings.allow_public_admin_registration:
        logger.warning(
            "Public registration attempted to create admin account; "
            "downgrading to 'user'. Set ALLOW_PUBLIC_ADMIN_REGISTRATION=true "
            "(dev only) to override."
        )
        return "user"
    return requested
