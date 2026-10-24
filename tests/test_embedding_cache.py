"""Unit tests for embedding vector caching (memory & MongoDB)."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.embeddings import EmbeddingEngine, compute_text_hash, reset_embedding_engine


def test_compute_text_hash():
    h1 = compute_text_hash("Sample chunk text", "BAAI/bge-m3")
    h2 = compute_text_hash("Sample chunk text", "BAAI/bge-m3")
    h3 = compute_text_hash("Sample chunk text ", "BAAI/bge-m3")
    h4 = compute_text_hash("Sample chunk text", "all-MiniLM-L6-v2")

    assert h1 == h2
    assert h1 == h3  # trailing whitespace stripped
    assert h1 != h4  # model name differs


def test_embedding_engine_mem_cache():
    async def _test():
        engine = EmbeddingEngine()
        engine.backend = "in_process"
        engine.model_name = "test-model-mem"

        mock_vec = [0.1, 0.2, 0.3]
        engine._encode_sync = MagicMock(return_value=[mock_vec])

        # Mock DB returning empty result so unit test is isolated from local MongoDB
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_db.embedding_cache.find.return_value = mock_cursor

        texts = ["Unique text for memory cache test only."]

        # First call — cache miss (calls _encode_sync)
        res1 = await engine.aget_embeddings(texts, db=mock_db)
        assert res1 == [mock_vec]
        assert engine._encode_sync.call_count == 1

        # Second call — cache hit from _mem_cache (should NOT call _encode_sync)
        res2 = await engine.aget_embeddings(texts, db=mock_db)
        assert res2 == [mock_vec]
        assert engine._encode_sync.call_count == 1  # count remains 1

    asyncio.run(_test())


def test_embedding_engine_mongo_cache_lookup():
    async def _test():
        engine = EmbeddingEngine()
        engine.backend = "in_process"
        engine.model_name = "test-model"

        mock_vec = [0.5, 0.6, 0.7]
        text = "Unique text chunk for Mongo cache test."
        text_hash = compute_text_hash(text, "test-model")

        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"hash": text_hash, "embedding": mock_vec}])
        mock_db.embedding_cache.find.return_value = mock_cursor

        engine._encode_sync = MagicMock(return_value=[[0.9, 0.9, 0.9]])

        # Call with db provided — should find it in DB cache
        res = await engine.aget_embeddings([text], db=mock_db)
        assert res == [mock_vec]
        # Model inference should not be called since DB returned cached vector
        assert engine._encode_sync.call_count == 0

    asyncio.run(_test())
