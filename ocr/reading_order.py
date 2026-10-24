"""Reading-order engine — multi-column aware ordering for legal PDFs.

Implements §4.4 of the spec. Designed to produce correct reading order on
Tamil multi-column legal documents where DocAI's default order can be off.

Algorithm
---------
1. Sort blocks by ``y`` (top → bottom).
2. Group blocks into ``rows`` — two blocks belong to the same row if their
   y-ranges overlap by more than ``ROW_OVERLAP_FRACTION`` of the smaller height.
3. Within each row, sort by ``x`` (left → right).
4. Detect columns: if a row's blocks are vertically tall and horizontally
   non-overlapping, treat them as separate columns; emit each column fully
   before moving to the next visual row band.
5. Assign sequential ``reading_order`` integers (0-based) to blocks in
   final order.
6. Detect section labels using bilingual legal regex patterns and write to
   ``block.section_label``.

The function is pure — input list is not mutated; new ``reading_order`` and
``section_label`` values are written on the returned blocks.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from .layout_schema import LayoutBlock


# ── Section-label detection ─────────────────────────────────────────────────

_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Tamil
    re.compile(r"^\s*(பிரிவு|விதி|உத்தரவு|பகுதி|அத்தியாயம்)\s+([\d০-৯௦-௯]+[\w()\.\-]*)"),
    # English legal numbering
    re.compile(r"^\s*(Rule|Section|Clause|Article|Order|Chapter|Schedule|Part)\s+([\dIVXLC]+[\w()\.\-]*)", re.IGNORECASE),
    # Bare numbered headings: "5.", "5.1", "5.1.2"
    re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.\s+\S"),
    # Tamil letter-numbered headings: அ., ஆ., இ.
    re.compile(r"^\s*([அ-ஹ])\.\s+\S"),
)


def detect_section_label(text: str) -> str | None:
    """Return the section/rule label parsed from the first line of ``text``.

    Returns ``None`` when no recognised pattern matches.
    """
    if not text:
        return None
    head = text.lstrip().splitlines()[0] if text.strip() else ""
    for pat in _SECTION_PATTERNS:
        m = pat.match(head)
        if m:
            # Build a normalised label
            groups = [g for g in m.groups() if g]
            return " ".join(groups).strip()
    return None


# ── Row / column ordering ───────────────────────────────────────────────────

#: Fraction of the smaller block height required for two blocks to share a row.
ROW_OVERLAP_FRACTION: float = 0.5


def recursive_xy_cut(blocks: List[LayoutBlock], horiz_gap: float = 4.0, vert_gap: float = 10.0) -> List[LayoutBlock]:
    """Sort blocks using recursive horizontal and vertical splits (Recursive XY-Cut).
    
    This handles arbitrary nested multi-column layouts, headers, footers, and
    marginalia with high visual fidelity.
    """
    if len(blocks) <= 1:
        return blocks

    # 1. Try to find a horizontal split (horizontal gap separating rows/bands)
    blocks_sorted_y = sorted(blocks, key=lambda b: b.y)
    split_idx = -1
    max_gap = -1
    
    current_max_y1 = blocks_sorted_y[0].y + max(blocks_sorted_y[0].height, 1.0)
    for i in range(len(blocks_sorted_y) - 1):
        b_curr = blocks_sorted_y[i]
        b_next = blocks_sorted_y[i+1]
        current_max_y1 = max(current_max_y1, b_curr.y + max(b_curr.height, 1.0))
        gap = b_next.y - current_max_y1
        if gap >= horiz_gap:
            if gap > max_gap:
                max_gap = gap
                split_idx = i

    if split_idx != -1:
        top_part = blocks_sorted_y[:split_idx+1]
        bottom_part = blocks_sorted_y[split_idx+1:]
        return recursive_xy_cut(top_part, horiz_gap, vert_gap) + recursive_xy_cut(bottom_part, horiz_gap, vert_gap)

    # 2. Try to find a vertical split (vertical gap separating side-by-side columns)
    blocks_sorted_x = sorted(blocks, key=lambda b: b.x)
    split_idx = -1
    max_gap = -1
    
    current_max_x1 = blocks_sorted_x[0].x + max(blocks_sorted_x[0].width, 1.0)
    for i in range(len(blocks_sorted_x) - 1):
        b_curr = blocks_sorted_x[i]
        b_next = blocks_sorted_x[i+1]
        current_max_x1 = max(current_max_x1, b_curr.x + max(b_curr.width, 1.0))
        gap = b_next.x - current_max_x1
        if gap >= vert_gap:
            if gap > max_gap:
                max_gap = gap
                split_idx = i

    if split_idx != -1:
        left_part = blocks_sorted_x[:split_idx+1]
        right_part = blocks_sorted_x[split_idx+1:]
        return recursive_xy_cut(left_part, horiz_gap, vert_gap) + recursive_xy_cut(right_part, horiz_gap, vert_gap)

    # Fallback: sort top-to-bottom, then left-to-right
    return sorted(blocks, key=lambda b: (b.y, b.x))


def assign_reading_order(blocks: Iterable[LayoutBlock]) -> List[LayoutBlock]:
    """Sort ``blocks`` into reading order and stamp ``reading_order`` on each.

    Returns the same block instances re-ordered. Uses the recursive XY-cut
    algorithm to handle columns and complex layout flows correctly.
    """
    items = list(blocks)
    if not items:
        return items

    # Sort blocks into reading order using recursive XY-cut
    ordered = recursive_xy_cut(items)

    # Stamp reading_order and detect section labels.
    for idx, blk in enumerate(ordered):
        blk.reading_order = idx
        if not blk.section_label:
            label = detect_section_label(blk.text)
            if label:
                blk.section_label = label

    return ordered


__all__ = ["assign_reading_order", "detect_section_label", "ROW_OVERLAP_FRACTION", "recursive_xy_cut"]

