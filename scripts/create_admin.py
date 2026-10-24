#!/usr/bin/env python3
"""Bootstrap an admin user from the command line.

Usage::

    python scripts/create_admin.py --email admin@example.com --password secret

The script connects directly to MongoDB using the same settings the
application uses, hashes the password with bcrypt, and inserts a user
with ``role='admin'``.

If the email already exists the script aborts so you don't accidentally
overwrite an existing account.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

# Allow running from repo root without installing the package
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

logger = logging.getLogger("create_admin")
logging.basicConfig(level=logging.INFO, format="%(message)s")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash(password: str) -> str:
    return pwd_context.hash(password[:72])


async def main() -> int:
    parser = argparse.ArgumentParser(description="Create an admin user")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="Admin password")
    parser.add_argument("--username", default="", help="Display name (defaults to email prefix)")
    args = parser.parse_args()

    username = args.username or args.email.split("@")[0]

    client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db_name]

    existing = await db.users.find_one({"email": args.email})
    if existing:
        logger.error("User already exists: %s", args.email)
        return 1

    doc = {
        "username": username,
        "email": args.email,
        "hashed_password": _hash(args.password),
        "role": "admin",
        "is_active": True,
        "created_at": __import__("datetime").datetime.utcnow(),
    }

    result = await db.users.insert_one(doc)
    logger.info(
        "Created admin user: %s (id=%s)",
        args.email,
        result.inserted_id,
    )

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
