from __future__ import annotations

import asyncio

from core.retrieval import RetrievalScope, _score_chunks_mongodb
from core.semantic_search import QueryProcessor, build_search_chunks


def test_property_boundary_fact_windows_are_created() -> None:
    chunks = build_search_chunks(
        doc_id="job-1",
        filename="sale-deed.pdf",
        text=(
            "--- Page 1 ---\n"
            "SCHEDULE OF PROPERTY\n"
            "Survey No. 123/4 measuring 1200 sq.ft.\n"
            "North by : Survey No. 122 land\n"
            "South by : Road\n"
        ),
    )

    fact_chunks = [c for c in chunks if c.metadata.get("chunk_type") in {"fact_window", "schedule"}]

    assert fact_chunks
    assert any("North by" in c.text and "Survey No" in c.text for c in fact_chunks)
    assert any("north" in c.metadata.get("field_hits", []) for c in fact_chunks)


def test_query_processor_detects_boundary_lookup() -> None:
    processed = QueryProcessor().process("North by : Survey No ?")

    assert processed.intent == "boundary_lookup"
    assert "north" in processed.key_terms
    assert "survey" in processed.key_terms


def test_lexical_fact_chunk_can_beat_unrelated_embedded_chunk(monkeypatch) -> None:
    class FakeEmbeddingEngine:
        async def aget_embedding(self, text: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr(
        "core.embeddings.get_embedding_engine",
        lambda: FakeEmbeddingEngine(),
    )

    processed = QueryProcessor().process("North by : Survey No ?")
    chunks = [
        {
            "job_id": "job-1",
            "filename": "sale-deed.pdf",
            "chunk_id": 1,
            "text": "This page discusses a special power of attorney.",
            "embedding": [1.0, 0.0],
            "metadata": {
                "filename": "sale-deed.pdf",
                "chunk_type": "semantic",
                "embedding_status": "ok",
            },
        },
        {
            "job_id": "job-1",
            "filename": "sale-deed.pdf",
            "chunk_id": 2,
            "text": "SCHEDULE OF PROPERTY\nSurvey No. 123/4\nNorth by : Survey No. 122 land",
            "embedding": [],
            "metadata": {
                "filename": "sale-deed.pdf",
                "chunk_type": "fact_window",
                "field_hits": ["survey_no", "north"],
                "embedding_status": "failed",
            },
        },
    ]

    result = asyncio.run(
        _score_chunks_mongodb(
            chunks,
            processed,
            top_k=2,
            threshold=0.0,
            and_terms=[],
            section_refs=[],
            scope=RetrievalScope(job_id="job-1"),
            retrieval_stages=["mongo_chunks"],
        )
    )

    assert result.matches[0].chunk_id == 2
    assert result.index_health == "degraded"
    assert result.fallback_used is True
    assert result.query_intent == "boundary_lookup"
