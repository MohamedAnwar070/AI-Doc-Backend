from __future__ import annotations

import asyncio

from core.semantic_search import (
    build_search_chunks,
    chunk_text,
    _split_sentences,
    _deduplicate_chunks,
    SearchChunk,
)
from voice.conversation_engine import VoiceAssistantResult, VoiceConversationEngine


# ── Sentence Splitting ─────────────────────────────────────────────────────

def test_split_sentences_basic() -> None:
    text = "This is sentence one. This is sentence two. And a third one!"
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "This is sentence one."
    assert sentences[1] == "This is sentence two."
    assert sentences[2] == "And a third one!"


def test_split_sentences_handles_abbreviations() -> None:
    text = "Dr. Smith signed the deed. Mr. Kumar was present. Survey No. 102 was verified."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert "Dr. Smith" in sentences[0]
    assert "Mr. Kumar" in sentences[1]
    assert "Survey No. 102" in sentences[2]


def test_split_sentences_preserves_decimal_numbers() -> None:
    text = "The area measures 1200.50 sq.ft. The rate is Rs.5,000 per sq.ft."
    sentences = _split_sentences(text)
    assert any("1200.50" in s for s in sentences)
    assert any("Rs.5,000" in s for s in sentences)


def test_split_sentences_handles_newlines_as_boundaries() -> None:
    text = "Line one\nLine two\nLine three"
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_handles_empty_input() -> None:
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


# ── Sentence-Aware Chunking ────────────────────────────────────────────────

def test_chunk_text_never_splits_mid_sentence() -> None:
    # Build text with clearly identifiable sentences totaling > 800 chars
    sentences = [
        f"Sentence number {i} of this legal document contains important factual information about the property."
        for i in range(1, 15)
    ]
    text = " ".join(sentences)
    assert len(text) > 800  # Ensure it's large enough to require chunking

    chunks = chunk_text(text, chunk_size=800, overlap=200)
    assert len(chunks) >= 2

    # Every chunk should end with a sentence-ending punctuation mark
    for chunk in chunks:
        stripped = chunk.strip()
        assert stripped[-1] in ".!?", f"Chunk ends with '{stripped[-5:]}' — not at sentence boundary"


def test_chunk_text_handles_long_single_sentence() -> None:
    # A single 2000-char sentence must be kept whole, never split mid-word
    long_sentence = "This is a very long sentence " * 80 + "and it ends here."
    assert len(long_sentence) > 2000

    chunks = chunk_text(long_sentence, chunk_size=800, overlap=200)
    assert len(chunks) == 1  # Single sentence → single chunk
    assert chunks[0] == long_sentence.strip()


def test_chunk_text_overlap_uses_sentence_boundaries() -> None:
    sentences = [
        "The vendor hereby sells the property.",
        "The vendee accepts the sale consideration.",
        "The property is bounded on the north by a road.",
        "The property is bounded on the south by Survey No 102.",
        "The total area is 1200 square feet.",
        "The sale consideration is Rs 10 lakhs.",
        "The deed was executed in the presence of witnesses.",
        "The Sub-Registrar verified all documents.",
    ]
    text = " ".join(sentences)

    chunks = chunk_text(text, chunk_size=300, overlap=100)
    assert len(chunks) >= 2

    # Verify overlap: the last sentence of chunk N should appear at the start of chunk N+1
    for i in range(len(chunks) - 1):
        this_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        # Find the last sentence of this chunk
        last_sent_of_this = [s for s in sentences if s in this_chunk][-1]
        # It should appear in the next chunk as overlap
        assert last_sent_of_this in next_chunk, (
            f"Overlap failed: '{last_sent_of_this[:40]}...' not in next chunk"
        )


def test_chunk_text_preserves_short_text_unchanged() -> None:
    short_text = "This deed of sale is executed in Chennai."
    chunks = chunk_text(short_text, chunk_size=800, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_text_preserves_paragraph_structure() -> None:
    text = (
        "First paragraph sentence one. First paragraph sentence two.\n\n"
        "Second paragraph sentence one. Second paragraph sentence two."
    )
    chunks = chunk_text(text, chunk_size=5000, overlap=200)
    assert len(chunks) == 1
    # Paragraph break should be preserved
    assert "\n\n" in chunks[0]


def test_chunk_text_empty_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ── Chunk Deduplication ────────────────────────────────────────────────────

def test_chunk_deduplication_removes_exact_duplicates() -> None:
    chunks = [
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=0, text="The property is in Velachery."),
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=1, text="The property is in Velachery."),
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=2, text="Different text entirely here."),
    ]
    deduped = _deduplicate_chunks(chunks)
    assert len(deduped) == 2
    assert deduped[0].chunk_id == 0
    assert deduped[1].chunk_id == 2


def test_chunk_deduplication_removes_near_duplicates() -> None:
    chunks = [
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=0,
                    text="The property is bounded on the north by a road and on the south by survey number one hundred two."),
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=1,
                    text="The property is bounded on the north by a road and on the south by survey number one hundred two"),
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=2,
                    text="Completely unrelated text about the execution date."),
    ]
    deduped = _deduplicate_chunks(chunks)
    assert len(deduped) == 2


def test_chunk_deduplication_preserves_distinct_chunks() -> None:
    chunks = [
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=0, text="Alpha bravo charlie delta echo."),
        SearchChunk(doc_id="d1", file_path="f1", chunk_id=1, text="Foxtrot golf hotel india juliet."),
    ]
    deduped = _deduplicate_chunks(chunks)
    assert len(deduped) == 2


# ── Neighbour Chunk Links ──────────────────────────────────────────────────

def test_neighbour_chunk_links() -> None:
    structured = {
        "pages": [
            {
                "page": 1,
                "elements": [
                    {"type": "paragraph", "text": "First paragraph about the property deed."},
                    {"type": "paragraph", "text": "Second paragraph about boundaries."},
                    {"type": "paragraph", "text": "Third paragraph about the consideration."},
                ],
            }
        ]
    }

    chunks = build_search_chunks(
        doc_id="job-links",
        filename="test.pdf",
        text="",
        structured=structured,
    )

    assert len(chunks) >= 2

    # First chunk in the finalized list should have no prev
    assert chunks[0].metadata["prev_chunk_id"] is None
    # Last chunk in the finalized list should have no next
    assert chunks[-1].metadata["next_chunk_id"] is None

    # Middle chunks should have both prev and next
    for i in range(1, len(chunks) - 1):
        assert chunks[i].metadata["prev_chunk_id"] is not None
        assert chunks[i].metadata["next_chunk_id"] is not None

    # All chunks should have the neighbour link keys
    for c in chunks:
        assert "prev_chunk_id" in c.metadata
        assert "next_chunk_id" in c.metadata


# ── Existing Tests (Phase 1 — must still pass) ────────────────────────────

def test_structured_chunks_embed_heading_context_without_polluting_text() -> None:
    structured = {
        "pages": [
            {
                "page": 2,
                "elements": [
                    {"type": "heading", "text": "Backend Technology"},
                    {
                        "type": "paragraph",
                        "text": "The platform uses FastAPI, MongoDB, and a React frontend.",
                    },
                ],
            }
        ]
    }

    chunks = build_search_chunks(
        doc_id="job-1",
        filename="architecture.pdf",
        text="",
        structured=structured,
    )

    paragraph = next(c for c in chunks if c.text.startswith("The platform uses"))

    assert paragraph.text == "The platform uses FastAPI, MongoDB, and a React frontend."
    assert "Heading: Backend Technology" in (paragraph.embed_text or "")
    assert "Document: architecture.pdf" in (paragraph.embed_text or "")
    assert paragraph.metadata["parent_heading"] == "Backend Technology"


def test_voice_with_document_scope_retrieves_before_llm_tool_decision(monkeypatch) -> None:
    from config import settings
    monkeypatch.setattr(settings, "voice_use_turn_pipeline", False)

    class ScopedVoiceEngine(VoiceConversationEngine):
        def __init__(self) -> None:
            super().__init__()
            self.search_called = False

        def _get_ai_client(self):  # pragma: no cover - should not be reached
            raise AssertionError("first-pass LLM should not run for scoped document turns")

        async def _search_documents(self, *args, **kwargs):
            self.search_called = True
            return "Page 1\nDocument context", [{"doc_id": "job-1", "chunk_id": 1}]

        async def _generate_contextual_answer(self, **kwargs):
            return VoiceAssistantResult(
                text="Document-grounded answer",
                sources=kwargs["sources"],
            )

    engine = ScopedVoiceEngine()

    result = asyncio.run(
        engine.generate_response_with_sources(
            user_text="what are the languages supported here?",
            doc_id="job-1",
            detected_language="en",
        )
    )

    assert engine.search_called is True
    assert result.text == "Document-grounded answer"
    assert result.sources == [{"doc_id": "job-1", "chunk_id": 1}]


def test_structured_chunks_multi_level_hierarchy() -> None:
    structured = {
        "pages": [
            {
                "page": 1,
                "elements": [
                    {"type": "heading", "text": "SCHEDULE OF PROPERTY"},
                    {"type": "heading", "text": "Item No. 1"},
                    {
                        "type": "paragraph",
                        "text": "All that piece and parcel of land situated in Velachery measuring 1200 sq.ft.",
                    },
                ],
            }
        ]
    }

    chunks = build_search_chunks(
        doc_id="job-hierarchy",
        filename="deed.pdf",
        text="",
        structured=structured,
    )

    land_chunk = next(c for c in chunks if c.text.startswith("All that piece"))
    assert land_chunk.metadata["hierarchy_path"] == "SCHEDULE OF PROPERTY > Item No. 1"
    assert land_chunk.metadata["section"] == "SCHEDULE OF PROPERTY"
    assert land_chunk.metadata["subsection"] == "Item No. 1"
    assert "Hierarchy: SCHEDULE OF PROPERTY > Item No. 1" in (land_chunk.embed_text or "")


def test_list_item_recognition() -> None:
    structured = {
        "pages": [
            {
                "page": 1,
                "elements": [
                    {"type": "paragraph", "text": "- 10% booking advance payable on execution"},
                    {"type": "paragraph", "text": "1. Balance payment within 30 days of notice"},
                    {"type": "paragraph", "text": "General non-list paragraph explaining normal rules."},
                ],
            }
        ]
    }

    chunks = build_search_chunks(
        doc_id="job-list",
        filename="terms.pdf",
        text="",
        structured=structured,
    )

    c1 = next(c for c in chunks if "10% booking" in c.text)
    c2 = next(c for c in chunks if "Balance payment" in c.text)
    c3 = next(c for c in chunks if "General non-list" in c.text)

    assert c1.metadata["chunk_type"] == "list_item"
    assert c1.metadata["element_type"] == "list_item"
    assert c2.metadata["chunk_type"] == "list_item"
    assert c2.metadata["element_type"] == "list_item"
    assert c3.metadata["chunk_type"] == "semantic"


# ── Phase 3: Parent-Child Chunking & Metadata Enrichment Tests ──────────────

def test_parent_child_chunking_and_linking() -> None:
    # Long text exceeding 1600 chars to trigger parent-child chunking
    sentences = [
        f"Paragraph sentence {i} contains detailed legal descriptions of the registered land parcel located in Tamil Nadu state."
        for i in range(1, 35)
    ]
    long_text = " ".join(sentences)
    assert len(long_text) > 2000

    chunks = build_search_chunks(
        doc_id="job-parent-child",
        filename="deed_long.pdf",
        text=long_text,
    )

    parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") != "parent"]

    assert len(parent_chunks) >= 1
    assert len(child_chunks) >= 2

    # Verify child chunks point to their parent_chunk_id and carry parent_text
    for child in child_chunks:
        if child.metadata.get("parent_chunk_id") is not None:
            p_id = child.metadata["parent_chunk_id"]
            p_chunk = next((p for p in parent_chunks if p.chunk_id == p_id), None)
            assert p_chunk is not None
            assert child.metadata["parent_text"] == p_chunk.text
            assert child.chunk_id in p_chunk.metadata["child_chunk_ids"]


def test_chunk_metadata_enrichment() -> None:
    text = "DEED OF SALE\n\nThis Sale Deed is executed in Chennai. கிரையப் பத்திரம்..."
    chunks = build_search_chunks(
        doc_id="job-meta",
        filename="sale_deed_doc.pdf",
        text=text,
    )

    assert len(chunks) >= 1
    c = chunks[0]
    assert c.metadata.get("language") in {"ta", "bilingual", "en"}
    assert c.metadata.get("title") is not None
    assert "DEED OF SALE" in c.metadata.get("title") or "Sale Deed" in c.metadata.get("title")
    assert "filename" in c.metadata



def test_context_formatter_uses_parent_text() -> None:
    from core.retrieval import MatchItem
    from core.context import format_chat_context, format_voice_context

    match = MatchItem(
        score=0.92,
        doc_id="job-1",
        chunk_id=5,
        text="Child chunk text excerpt.",
        filename="doc.pdf",
        page=2,
        metadata={
            "parent_text": "Full parent text containing the complete paragraph with Child chunk text excerpt inside it."
        },
    )

    chat_ctx = format_chat_context([match])
    assert "Full parent text containing the complete paragraph" in chat_ctx

    voice_ctx, sources = format_voice_context([match])
    assert "Full parent text containing the complete paragraph" in voice_ctx
    assert sources[0]["text"] == "Child chunk text excerpt."


