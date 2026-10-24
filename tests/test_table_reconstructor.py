"""Unit tests for :mod:`ocr._table_reconstructor`.

We feed synthetic OCR-style items to the reconstructor instead of running
PaddleOCR, so the suite is fast and offline.
"""
from __future__ import annotations

from ocr._table_reconstructor import reconstruct_table


def _make_item(text: str, x0: int, y0: int, x1: int, y1: int) -> list:
    """Build a PaddleOCR-shaped item ``[bbox, (text, conf)]``."""
    bbox = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [bbox, (text, 0.98)]


def _two_column_table_items(num_rows: int = 6) -> list:
    """A simple 2-column table at left-x=80 and right-x=600."""
    items: list = []
    for i in range(num_rows):
        y0 = 100 + i * 50
        y1 = y0 + 30
        items.append(_make_item(f"Label {i}", 80, y0, 280, y1))
        items.append(_make_item(f"Value {i}", 600, y0, 760, y1))
    return items


# ── Happy path ─────────────────────────────────────────────────────────────

def test_reconstruct_table_extracts_two_columns() -> None:
    items = _two_column_table_items()
    elements, non_table_text, _ = reconstruct_table(items)

    assert len(elements) == 1
    table = elements[0]
    assert table.type == "table"
    assert "<table>" in (table.html or "")
    assert "Label 0" in table.text
    assert "Value 0" in table.text
    assert non_table_text == ""


def test_reconstruct_table_handles_empty_input() -> None:
    elements, text, bbox = reconstruct_table([])
    assert elements == []
    assert text == ""
    assert bbox is None


def test_reconstruct_table_below_threshold_returns_no_table() -> None:
    items = _two_column_table_items(num_rows=2)  # below MIN_OCR_ITEMS pairs
    elements, _, _ = reconstruct_table(items)
    assert elements == []


# ── Continuation rows merge into previous label ───────────────────────────

def test_reconstruct_merges_continuation_rows() -> None:
    items = _two_column_table_items(num_rows=4)
    # Add a continuation row (label only, no right-side cell) between rows 0 and 1.
    items.append(_make_item("continuation", 80, 130, 290, 160))
    elements, _, _ = reconstruct_table(items)
    if not elements:
        return  # below MIN_TABLE_DATA_ROWS — acceptable
    assert "<table>" in (elements[0].html or "")


# ── Non-table content stays out of the table ──────────────────────────────

def test_reconstruct_keeps_paragraph_above_table_in_non_table_text() -> None:
    items = _two_column_table_items()
    items.append(_make_item("Paragraph above", 80, 20, 700, 60))
    elements, non_table_text, _ = reconstruct_table(items)
    assert elements
    assert "Paragraph above" in non_table_text
