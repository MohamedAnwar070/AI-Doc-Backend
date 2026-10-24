"""JWT token creation and verification.

Centralised so every code-path that mints or validates a JWT goes
through one audited place. Two small but production-critical
behaviours live here:

* ``timezone.utc`` for ``exp`` so tokens are valid no matter what
  the server's local clock thinks. ``datetime.utcnow()`` is naive
  and is being deprecated in Python 3.12+.
* ``_assert_secret_is_safe`` runs once at import time and refuses
  to start the process if the default secret leaks into anything
  but a development build.
"""
from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from config import settings

logger = logging.getLogger(__name__)


_DEFAULT_SECRETS = frozenset({"super-secret-key", "change-me", "secret"})


def _assert_secret_is_safe() -> None:
    """Refuse to mint tokens with a known-default secret in production.

    We treat ``APP_ENV=production`` as the "must be hardened" trigger.
    Anywhere else (dev/test/CI) we just warn loudly so developers get
    a heads-up but their workflow keeps moving.
    """
    secret = (settings.jwt_secret_key or "").strip()
    if not secret or secret in _DEFAULT_SECRETS:
        env = (os.getenv("APP_ENV") or "").lower()
        message = (
            "JWT_SECRET_KEY is using a known default value. Generate a "
            "strong random secret (e.g. `openssl rand -hex 32`) and set "
            "JWT_SECRET_KEY in your environment."
        )
        if env == "production":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        logger.warning(message)


_assert_secret_is_safe()


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create and sign a short-lived access token.

    ``data`` is merged with an ``exp`` claim. Callers should put the
    user id under ``sub`` and any role/claim data they need under
    additional keys.
    """
    to_encode = dict(data)
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> dict:
    """Decode and verify a JWT.

    Raises ``jwt.PyJWTError`` (or a subclass — ``ExpiredSignatureError``,
    ``InvalidTokenError``, etc.) on any failure.
    """
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
