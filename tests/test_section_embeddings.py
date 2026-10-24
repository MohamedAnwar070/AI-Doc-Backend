"""Unit tests for section chunk generation and parent/section embedding options."""

import pytest
from core.semantic_search import build_search_chunks


def test_section_chunk_generation():
    sample_text = """
    --- Page 1 ---
    Section 5: Registration Fees and Charges
    The registration fee for this document is paid as per regulations.
    The valuation is calculated based on market rate.
    """

    chunks = build_search_chunks(
        doc_id="test_job_1",
        filename="test_deed.pdf",
        text=sample_text,
    )

    chunk_types = [c.metadata.get("chunk_type") for c in chunks]
    assert "section" in chunk_types

    section_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "section"]
    assert len(section_chunks) > 0
    assert "SECTION SUMMARY" in section_chunks[0].text or "Section 5" in section_chunks[0].text
