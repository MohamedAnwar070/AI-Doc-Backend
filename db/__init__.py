"""Database package — MongoDB motor client."""
from .mongo import get_db, db_client

__all__ = ["get_db", "db_client"]
