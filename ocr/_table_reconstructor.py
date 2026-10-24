"""Spatial 2-column table reconstruction for flat OCR output.

Extracted from the original :mod:`ocr.paddle_engine` to keep that module
focused on engine plumbing and to drop its cognitive complexity below
SonarQube's threshold. All logic in this module is **pure** (no I/O,
no engine state) and operates on parsed OCR items.

Algorithm overview
------------------
1. Group raw OCR items into rows by y-coordinate proximity.
2. Phase 1 — locate the **rightmost consistent column** by clustering the
   left edges of right-side items that follow a wide gap.
3. Phase 2 — collect rows whose right-side item lines up with that column.
4. Phase 3 — pull a header row in if the row above has a single right-side
   token at the detected column.
5. Phase 4 — split each row into ``c0`` (left/label) and ``c1`` (right/value).
6. Phase 5 — merge continuation rows whose ``c1`` is empty into the
   previous row's ``c0``.
7. Phase 5.5 — rescue date / number cells using a separate English-only
   OCR pass (Paddle's ``ta`` model garbles digits).
8. Phase 6-8 — emit HTML + plain-text + bbox metadata.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from core.layout_engine import PageElement

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────

MIN_OCR_ITEMS = 6
MIN_ROWS = 4
GAP_FOR_CANDIDATE = 60
MAX_ITEM_WIDTH = 350
CLUSTER_DIST = 60
MIN_SUPPORT = 4
COLUMN_TOLERANCE = 35
MIN_TABLE_GAP = 30
MIN_TABLE_DATA_ROWS = 3
LEFT_TOLERANCE = 60
HEADER_DENSITY_MAX = 0.6
COL_BOUNDARY_OFFSET = 30


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class _OcrItem:
    text: str
    xl: float
    xr: float
    yt: float
    yb: float
    yc: float
    h: float


@dataclass
class _MergedRow:
    c0: str
    c1: str
    c0_items: list[_OcrItem]
    c1_items: list[_OcrItem]


# ── Public entry point ──────────────────────────────────────────────────────

def reconstruct_table(
    ocr_items: Sequence,
    en_ocr_items: Optional[Sequence] = None,
) -> tuple[list[PageElement], str, Optional[list[int]]]:
    """Detect a 2-column table and reconstruct it from spatial OCR boxes.

    Parameters
    ----------
    ocr_items:
        Raw PaddleOCR result items shaped ``[bbox, (text, confidence)]``.
    en_ocr_items:
        Optional English-only OCR pass over the same image. Used to repair
        date / number cells that the Tamil model garbles. Pass ``None`` to
        skip date repair.

    Returns
    -------
    ``(elements, non_table_text, non_table_bbox)``
        Where ``elements`` is a list with at most one ``table``
        :class:`PageElement` (empty if no table was found),
        ``non_table_text`` contains every line outside the table,
        and ``non_table_bbox`` is the bounding box around it (or ``None``).
    """
    parsed = _parse_items(ocr_items)
    if len(parsed) < MIN_OCR_ITEMS:
        return [], "\n".join(_lines_from_raw(ocr_items)), None

    rows = _group_into_rows(parsed)
    if len(rows) < MIN_ROWS:
        return [], "\n".join(it.text for it in parsed), None

    right_col_x = _detect_right_column(rows, parsed)
    if right_col_x is None:
        return [], "\n".join(_lines_from_raw(ocr_items)), None

    table_row_indices = _gather_table_rows(rows, right_col_x)
    if len(table_row_indices) < MIN_TABLE_DATA_ROWS:
        return [], "\n".join(_lines_from_raw(ocr_items)), None

    table_row_indices = _attach_header_row(rows, table_row_indices, right_col_x)

    merged = _split_and_merge_rows(rows, table_row_indices, right_col_x)
    if len(merged) < 2:
        return [], "\n".join(_lines_from_raw(ocr_items)), None

    if en_ocr_items:
        _repair_dates_with_english_ocr(merged, en_ocr_items, right_col_x)

    table_element = _build_table_element(merged, rows, table_row_indices)
    non_table_text, non_table_bbox = _extract_non_table_text(rows, table_row_indices)
    return [table_element], non_table_text, non_table_bbox


# ── Phase 0: parsing ────────────────────────────────────────────────────────

def _parse_items(ocr_items: Sequence) -> list[_OcrItem]:
    out: list[_OcrItem] = []
    for raw in ocr_items:
        if not raw or len(raw) < 2:
            continue
        bbox = raw[0]
        text_part = raw[1]
        text = text_part[0] if isinstance(text_part, (list, tuple)) else str(text_part)
        text = (text or "").strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        out.append(
            _OcrItem(
                text=text,
                xl=min(xs),
                xr=max(xs),
                yt=min(ys),
                yb=max(ys),
                yc=(min(ys) + max(ys)) / 2.0,
                h=max(ys) - min(ys),
            )
        )
    return out


def _lines_from_raw(ocr_items: Sequence) -> list[str]:
    """Cheap fallback line reconstruction when table detection bails out."""
    parsed = _parse_items(ocr_items)
    rows = _group_into_rows(parsed)
    return [" ".join(item.text for item in row) for row in rows if row]


# ── Phase 0.5: row grouping ────────────────────────────────────────────────

def _group_into_rows(items: list[_OcrItem]) -> list[list[_OcrItem]]:
    if not items:
        return []
    items_sorted = sorted(items, key=lambda i: i.yc)
    heights = sorted(i.h for i in items_sorted)
    median_h = heights[len(heights) // 2] if heights else 20.0
    y_threshold = max(median_h * 0.6, 10.0)

    rows: list[list[_OcrItem]] = []
    current_row = [items_sorted[0]]
    current_y = items_sorted[0].yc
    for item in items_sorted[1:]:
        if abs(item.yc - current_y) <= y_threshold:
            current_row.append(item)
            current_y = (current_y + item.yc) / 2.0
        else:
            rows.append(sorted(current_row, key=lambda i: i.xl))
            current_row = [item]
            current_y = item.yc
    rows.append(sorted(current_row, key=lambda i: i.xl))
    return rows


# ── Phase 1: rightmost column detection ────────────────────────────────────

def _detect_right_column(
    rows: list[list[_OcrItem]],
    parsed: list[_OcrItem],
) -> Optional[float]:
    page_right = max(it.xr for it in parsed)
    page_left = min(it.xl for it in parsed)
    page_center_x = (page_left + page_right) / 2.0

    candidates: list[float] = []
    for row in rows:
        for ji, it in enumerate(row):
            if it.xl < page_center_x:
                continue
            if (it.xr - it.xl) > MAX_ITEM_WIDTH:
                continue
            if ji == 0:
                continue
            left_gap = it.xl - row[ji - 1].xr
            if left_gap >= GAP_FOR_CANDIDATE:
                candidates.append(it.xl)

    if len(candidates) < MIN_SUPPORT:
        return None

    xls_sorted = sorted(set(candidates))
    clusters: list[list[float]] = [[xls_sorted[0]]]
    for xl in xls_sorted[1:]:
        if xl - clusters[-1][-1] < CLUSTER_DIST:
            clusters[-1].append(xl)
        else:
            clusters.append([xl])

    best_avg: Optional[float] = None
    best_count = 0
    for cluster in clusters:
        avg = sum(cluster) / len(cluster)
        count = sum(1 for xl in candidates if abs(xl - avg) < CLUSTER_DIST)
        if count > best_count:
            best_count = count
            best_avg = avg

    if best_avg is None or best_count < MIN_SUPPORT:
        return None
    logger.debug("Right column at xl≈%.0f (%d supporting rows)", best_avg, best_count)
    return best_avg


# ── Phase 2: table row gathering ────────────────────────────────────────────

def _gather_table_rows(
    rows: list[list[_OcrItem]],
    right_col_x: float,
) -> set[int]:
    table_data_rows: set[int] = set()
    for ri, row in enumerate(rows):
        for ji, it in enumerate(row):
            if abs(it.xl - right_col_x) > COLUMN_TOLERANCE:
                continue
            if (it.xr - it.xl) > MAX_ITEM_WIDTH:
                continue
            if ji > 0 and (it.xl - row[ji - 1].xr) >= MIN_TABLE_GAP:
                table_data_rows.add(ri)
                break

    if len(table_data_rows) < MIN_TABLE_DATA_ROWS:
        return table_data_rows

    table_left_xls = [rows[ri][0].xl for ri in table_data_rows]
    table_left_avg = sum(table_left_xls) / len(table_left_xls)

    min_tr = min(table_data_rows)
    max_tr = max(table_data_rows)
    all_table_rows: set[int] = set(table_data_rows)
    for ri in range(min_tr, max_tr + 1):
        if ri not in all_table_rows and rows[ri]:
            if abs(rows[ri][0].xl - table_left_avg) < LEFT_TOLERANCE:
                all_table_rows.add(ri)

    for ri in range(max_tr + 1, len(rows)):
        if rows[ri] and abs(rows[ri][0].xl - table_left_avg) < LEFT_TOLERANCE:
            all_table_rows.add(ri)
        else:
            break
    return all_table_rows


def _attach_header_row(
    rows: list[list[_OcrItem]],
    table_rows: set[int],
    right_col_x: float,
) -> set[int]:
    if not table_rows:
        return table_rows
    min_tr = min(table_rows)
    if min_tr == 0:
        return table_rows
    header_row = rows[min_tr - 1]
    if len(header_row) < 2:
        return table_rows
    total_span = header_row[-1].xr - header_row[0].xl
    text_width = sum(it.xr - it.xl for it in header_row)
    if total_span <= 0 or (text_width / total_span) >= HEADER_DENSITY_MAX:
        return table_rows
    page_center_threshold = right_col_x - GAP_FOR_CANDIDATE
    if any(it.xl >= page_center_threshold for it in header_row):
        return table_rows | {min_tr - 1}
    return table_rows


# ── Phase 4-5: row splitting + continuation merging ────────────────────────

def _split_and_merge_rows(
    rows: list[list[_OcrItem]],
    table_row_indices: set[int],
    right_col_x: float,
) -> list[_MergedRow]:
    col_boundary = right_col_x - COL_BOUNDARY_OFFSET
    sorted_indices = sorted(table_row_indices)

    raw: list[_MergedRow] = []
    for ri in sorted_indices:
        c0_items: list[_OcrItem] = []
        c1_items: list[_OcrItem] = []
        for it in rows[ri]:
            (c1_items if it.xl >= col_boundary else c0_items).append(it)
        raw.append(
            _MergedRow(
                c0=" ".join(i.text for i in c0_items),
                c1=" ".join(i.text for i in c1_items),
                c0_items=c0_items,
                c1_items=c1_items,
            )
        )

    merged: list[_MergedRow] = []
    for row in raw:
        if merged and not row.c1.strip():
            merged[-1].c0 += " " + row.c0
            merged[-1].c0_items.extend(row.c0_items)
        else:
            merged.append(
                _MergedRow(
                    c0=row.c0,
                    c1=row.c1,
                    c0_items=list(row.c0_items),
                    c1_items=list(row.c1_items),
                )
            )
    for m in merged:
        m.c0 = m.c0.strip()
        m.c1 = m.c1.strip()
    return merged


# ── Phase 5.5: date repair using English-only OCR pass ─────────────────────

_DATE_RE = re.compile(r"\d{1,2}\.\d{2}\.\d{4}")
_TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")
_NON_DATE_RE = re.compile(r"[^\d.\s]")
_DATE_SHAPE_RE = re.compile(r"^(\d{1,2})\.(\d{2})\.(\d{4})$")
_DOT_SPACE_RE = re.compile(r"[.\s]")


def _extract_date(text: str) -> Optional[str]:
    match = _DATE_RE.search(text)
    return match.group(0) if match else None


def _normalize_date(text: str) -> Optional[str]:
    cleaned = _TAMIL_CHAR_RE.sub("", text).strip()
    cleaned = _NON_DATE_RE.sub("", cleaned).strip()
    if _DATE_SHAPE_RE.match(cleaned):
        return cleaned
    digits_only = _DOT_SPACE_RE.sub("", cleaned)
    if len(digits_only) == 8 and digits_only.isdigit():
        dd, mm, yyyy = digits_only[:2], digits_only[2:4], digits_only[4:]
        try:
            if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12 and 2000 <= int(yyyy) <= 2030:
                return f"{dd}.{mm}.{yyyy}"
        except ValueError:
            return None
    return None


def _repair_dates_with_english_ocr(
    merged: list[_MergedRow],
    en_ocr_items: Sequence,
    right_col_x: float,
) -> None:
    en_parsed = _parse_items(en_ocr_items)
    col_boundary = right_col_x - COL_BOUNDARY_OFFSET
    repaired = 0

    for row in merged:
        if not row.c1_items or not any(c.isdigit() for c in row.c1):
            continue
        yt = min(i.yt for i in row.c1_items)
        yb = max(i.yb for i in row.c1_items)

        if _try_direct_en_match(row, en_parsed, col_boundary, yt, yb):
            repaired += 1
            continue
        if _try_overlap_en_match(row, en_parsed, col_boundary, yt, yb):
            repaired += 1
            continue
        normalized = _normalize_date(row.c1)
        if normalized and normalized != row.c1:
            row.c1 = normalized
            repaired += 1

    if repaired:
        logger.info("Date repair: corrected %d cells", repaired)


def _try_direct_en_match(
    row: _MergedRow,
    en_parsed: list[_OcrItem],
    col_boundary: float,
    yt: float,
    yb: float,
) -> bool:
    direct = sorted(
        [
            e
            for e in en_parsed
            if e.xl >= col_boundary - 50
            and (e.xr - e.xl) < 400
            and yt - 50 <= e.yc <= yb + 50
        ],
        key=lambda e: e.xl,
    )
    if not direct:
        return False
    en_text = " ".join(e.text for e in direct)
    if not any(c.isdigit() for c in en_text):
        return False
    row.c1 = en_text
    return True


def _try_overlap_en_match(
    row: _MergedRow,
    en_parsed: list[_OcrItem],
    col_boundary: float,
    yt: float,
    yb: float,
) -> bool:
    overlap = [
        e
        for e in en_parsed
        if e.xr >= col_boundary + 30
        and e.xl < col_boundary - 50
        and yt - 15 <= e.yc <= yb + 15
    ]
    for candidate in overlap:
        extracted = _extract_date(candidate.text)
        if extracted:
            row.c1 = extracted
            return True
    return False


# ── Phase 6-8: HTML / text / bbox emission ─────────────────────────────────

def _build_table_element(
    merged: list[_MergedRow],
    rows: list[list[_OcrItem]],
    table_row_indices: set[int],
) -> PageElement:
    final_rows = [[m.c0, m.c1] for m in merged]
    parts = ["<html><body><table><thead><tr>"]
    for cell in final_rows[0]:
        parts.append(f"<td>{cell}</td>")
    parts.append("</tr></thead><tbody>")
    for row in final_rows[1:]:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></body></html>")
    table_html = "".join(parts)
    table_text = "\n".join("\t".join(row) for row in final_rows)

    table_items = [it for ri in table_row_indices for it in rows[ri]]
    bbox: Optional[list[int]] = None
    if table_items:
        bbox = [
            round(min(it.xl for it in table_items)),
            round(min(it.yt for it in table_items)),
            round(max(it.xr for it in table_items)),
            round(max(it.yb for it in table_items)),
        ]
    logger.info("Table reconstructed: %d rows × 2 cols (bbox=%s)", len(final_rows), bbox)
    return PageElement(type="table", text=table_text, html=table_html, bbox=bbox)


def _extract_non_table_text(
    rows: list[list[_OcrItem]],
    table_row_indices: set[int],
) -> tuple[str, Optional[list[int]]]:
    non_table_items = [
        it for ri in range(len(rows)) if ri not in table_row_indices for it in rows[ri]
    ]
    bbox: Optional[list[int]] = None
    if non_table_items:
        bbox = [
            round(min(it.xl for it in non_table_items)),
            round(min(it.yt for it in non_table_items)),
            round(max(it.xr for it in non_table_items)),
            round(max(it.yb for it in non_table_items)),
        ]
    parts = [
        " ".join(it.text for it in rows[ri])
        for ri in range(len(rows))
        if ri not in table_row_indices
    ]
    return "\n".join(parts), bbox
