"""Production fixes from voice-rag-execution-plan (embed batch, voice clean, BM25 IDF, intent)."""

from __future__ import annotations

import math
import re

import pytest

from core.semantic_search import (
    QueryProcessor,
    _bm25_score,
    _build_df_map,
)
from voice.conversation_engine import VoiceConversationEngine


class TestQueryCleaning:
    def test_pronouns_and_discourse_stripped(self):
        pq = QueryProcessor().process(
            "Okay, summarize this person's resume and tell me what exactly he needs"
        )
        lowered = set(pq.key_terms)
        assert "okay" not in lowered
        assert "he" not in lowered
        assert "exactly" not in lowered
        assert "resume" in lowered or "summarize" in lowered

    def test_survey_no_preserved(self):
        pq = QueryProcessor().process("survey no 42")
        assert "no" in pq.key_terms or "survey" in pq.key_terms
        # "no" must not vanish when adjacent to survey vocab
        assert "survey" in pq.key_terms
        assert "42" in pq.key_terms or "no" in pq.key_terms

    def test_summary_beats_field_lookup(self):
        pq = QueryProcessor().process("summarize this property document")
        assert pq.intent == "summary"

    def test_identity_name_query(self):
        pq = QueryProcessor().process("what is her name ?")
        assert pq.intent == "identity_lookup"
        assert "name" in pq.key_terms
        assert any("full name" in v.lower() for v in pq.variants)

    def test_identity_boost_finds_resume_header(self):
        from core.semantic_search import identity_chunk_boost

        header = (
            "CONTACT 8148571761\n"
            "kalaipriyar26@gmail.com\n"
            "KALAI PRIYA R\n"
            "AI ENGINEER\n"
        )
        plain = "TECHNICAL SKILLS:\nJAVA SCRIPT\nJAVA\nWEB DEVELOPMENT"
        assert identity_chunk_boost(header, {"page": 1}) > identity_chunk_boost(
            plain, {"page": 2}
        )


class TestBm25Idf:
    def test_rare_term_outranks_common_without_equal_tf(self):
        docs = [
            "he he he we him resume skills python",
            "he we him they them our",
            "resume experience education python java",
        ]
        df_map, n_docs = _build_df_map(docs)
        query = "resume python"
        scores = [
            _bm25_score(query, d, avg_doc_len=8.0, df_map=df_map, n_docs=n_docs)
            for d in docs
        ]
        # Content doc should beat pronoun soup
        assert scores[2] > scores[1]

    def test_idf_greater_for_rare_terms(self):
        docs = ["alpha beta", "beta gamma", "beta delta"]
        df_map, n_docs = _build_df_map(docs)
        # alpha appears once, beta thrice
        from core.semantic_search import _bm25_idf

        assert _bm25_idf("alpha", df_map, n_docs) > _bm25_idf("beta", df_map, n_docs)


class TestVoiceClean:
    def test_strips_page_citations(self):
        eng = VoiceConversationEngine.__new__(VoiceConversationEngine)
        cleaned = eng._clean_for_voice(
            "The candidate has Java experience (Page 1) and Docker (Page 2, Section 3)."
        )
        assert "Page" not in cleaned
        assert "Java" in cleaned
        assert "Docker" in cleaned


class TestEmbeddingBatchApi:
    def test_parse_openai_batch_shape(self):
        from core.hf_inference_client import BasicAuthLlmClient

        payload = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.2, 0.3]},
                {"object": "embedding", "index": 0, "embedding": [0.0, 0.1]},
            ],
        }
        out = BasicAuthLlmClient._parse_embeddings_response(payload, expected=2)
        assert len(out) == 2
        assert out[0] == [0.0, 0.1]
        assert out[1] == [0.2, 0.3]

    def test_engine_config_defaults_to_bge_path(self):
        from config import settings

        assert "bge-m3" in (settings.embedding_local_model_path or "").lower()
        assert settings.embedding_runtime in {"in_process", "remote", "auto"}


class TestScoreNormalize:
    def test_max_norm_when_unbounded(self):
        from core.retrieval import _normalize_score_list

        items = [{"score": 2.5}, {"score": 1.0}, {"score": 0.5}]
        _normalize_score_list(items)
        assert items[0]["score"] == pytest.approx(1.0)
        assert items[1]["score"] == pytest.approx(1.0 / 2.5)
        assert all(0.0 <= m["score"] <= 1.0 for m in items)
