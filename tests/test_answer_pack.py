"""Tests for match selection and voice clean helpers."""

from core.answer_pack import (
    clean_for_voice,
    enforce_single_doc,
    is_refusal_answer,
    query_evidence_in_matches,
    select_matches_for_answer,
)
from core.intent import resolve_turn_intent
from core.retrieval import MatchItem, RetrievalResult


def _match(score, text, chunk_id=0, doc_id="d1"):
    return MatchItem(
        score=score,
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        file_path="f",
        filename="f.pdf",
        page=1,
        source="structured",
    )


def test_soft_gate_single_doc():
    intent = resolve_turn_intent("scanned with ?", channel="chat", has_doc_scope=True)
    matches = [
        _match(0.2, "Scanned with OKEN Scanner", 1),
        _match(0.1, "other text", 2),
    ]
    result = RetrievalResult(
        matches=matches,
        relevant_matches=[],  # all under threshold
        chunks_evaluated=2,
        retrieval_mode="mongodb",
        and_terms=[],
        section_refs=[],
        exact_section_match_found=False,
        confidence="low",
        scope_label="single_doc",
        query_intent="keyword",
    )
    selected, soft = select_matches_for_answer(
        result,
        intent,
        modality="chat",
        single_doc_only=True,
        requested_doc_id="d1",
    )
    assert soft is True
    assert len(selected) == 2
    assert "OKEN" in selected[0].text


def test_evidence_and_refusal():
    m = _match(1.0, "Scanned with OKEN Scanner")
    assert query_evidence_in_matches("scanned with ?", [m])
    assert is_refusal_answer(
        "This information is not available — please contact the Sub-Registrar."
    )
    assert not is_refusal_answer("It was scanned with OKEN Scanner.")


def test_clean_for_voice_strips_page():
    cleaned = clean_for_voice("Answer here (Page 3, Section 2).")
    assert "Page" not in cleaned
    assert "Answer here" in cleaned


def test_enforce_single_doc():
    items = [_match(1.0, "a", doc_id="x"), _match(0.5, "b", doc_id="y")]
    out = enforce_single_doc(
        items, requested_doc_id="y", scope_label="all_docs", single_doc_only=True
    )
    assert len(out) == 1 and out[0].doc_id == "y"
