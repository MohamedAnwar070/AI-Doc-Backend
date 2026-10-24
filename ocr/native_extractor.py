"""Native PDF extractor using pdfplumber.

Implements §4.2 of the spec. For PDFs classified as ``native`` by
:mod:`ocr.pdf_detector`, this module produces structured per-page output
WITHOUT calling Google Document AI — saving cost and latency.

Output format
-------------
Each page is returned as the same dict shape that the OCR worker already
persists (``PageResult.to_dict()``)::

    {"page": N, "elements": [
        {"type": "heading|paragraph|table",
         "text": "...",
         "html": null,
         "bbox": [x0, y0, x1, y1],   # PIXEL coords at DEFAULT_DPI
         "language": null,
         "confidence": 1.0},
        ...
    ]}

The pixel coords are produced by scaling pdfplumber's PDF-point coordinates
by ``DEFAULT_DPI / 72`` so the layout builder treats them like any other
engine's pixel bbox.

Heuristics
----------
* Heading detection: a text line whose font size is in the top 10% of the
  page's distribution AND is at most one line tall is a heading.
* Tables: pdfplumber ``page.extract_tables()`` — each table becomes one
  ``"table"`` element with HTML content; cells are not currently exposed
  individually (kept simple — Phase 0 priority is text correctness).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


PIXELS_PER_POINT: float = 300.0 / 72.0  # DEFAULT_DPI / 72


class NativeExtractionError(RuntimeError):
    pass


def _to_pixels(bbox: tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [
        round(x0 * PIXELS_PER_POINT, 2),
        round(y0 * PIXELS_PER_POINT, 2),
        round(x1 * PIXELS_PER_POINT, 2),
        round(y1 * PIXELS_PER_POINT, 2),
    ]


def _classify_line(font_size: float, body_size: float) -> str:
    """Return ``'heading'`` when font size is clearly larger than body."""
    if body_size > 0 and font_size >= body_size * 1.25:
        return "heading"
    if font_size >= 16:
        return "heading"
    return "paragraph"


def _table_to_html(table: list[list[Optional[str]]]) -> str:
    rows: list[str] = []
    for row in table:
        cells = "".join(
            f"<td>{(cell or '').strip()}</td>" for cell in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table>{''.join(rows)}</table>"


def _is_real_table(table: Any) -> bool:
    """Validate if a pdfplumber-detected table is a real tabular grid.
    
    Rejects visual sidebar alignment layouts and background color boxes which
    are falsely detected as sparse tables.
    """
    try:
        data = table.extract()
        if not data:
            return False
        num_rows = len(data)
        num_cols = len(data[0]) if num_rows > 0 else 0
        
        # A real table needs multiple columns and rows to be meaningful
        if num_rows <= 1 or num_cols <= 1:
            return False
            
        total_cells = num_rows * num_cols
        filled_cells = sum(1 for row in data for cell in row if cell and cell.strip())
        sparsity_ratio = filled_cells / total_cells
        
        # A real table must have a dense grid. Sparsity < 25% is almost
        # certainly a visual layout container/column alignment wrapper.
        if sparsity_ratio < 0.25:
            return False
            
        return True
    except Exception:
        return False


def _extract_page(page: Any, page_num: int) -> dict[str, Any]:
    """Extract one pdfplumber Page → page_dict, with high layout alignment."""
    elements: list[dict[str, Any]] = []

    # ── Tables first ────────────────────────────────────────────────────────
    try:
        tables = page.find_tables()
    except Exception:
        tables = []

    table_bboxes: list[tuple[float, float, float, float]] = []
    for table in tables or []:
        if not _is_real_table(table):
            continue
        try:
            data = table.extract()
            if not data:
                continue
            html = _table_to_html(data)
            text_lines = [
                "\t".join((cell or "").strip() for cell in row)
                for row in data
            ]
            elements.append(
                {
                    "type": "table",
                    "text": "\n".join(text_lines).strip(),
                    "html": html,
                    "bbox_pts": tuple(table.bbox),  # store PDF points for sorting
                    "language": None,
                    "confidence": 1.0,
                }
            )
            table_bboxes.append(tuple(table.bbox))
        except Exception as exc:
            logger.debug("Table extraction failed on page %d: %s", page_num, exc)

    # ── Body text grouped into lines via pdfplumber utilities ───────────────
    try:
        # We need size and fontname to identify headings vs body text
        words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["size", "fontname"],
        )
    except Exception:
        words = []

    if not words:
        # Fallback: take entire page text as one paragraph if no words found but page has some text
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            elements.append(
                {
                    "type": "paragraph",
                    "text": text,
                    "html": None,
                    "bbox_pts": (float(page.bbox[0]), float(page.bbox[1]),
                                 float(page.bbox[2]), float(page.bbox[3])),
                    "language": None,
                    "confidence": 1.0,
                }
            )
        # Scale to pixels and return
        for el in elements:
            if "bbox_pts" in el:
                el["bbox"] = _to_pixels(el.pop("bbox_pts"))
        return {"page": page_num, "elements": elements}

    # Group words into lines using y-midpoint binning
    sizes = [float(w.get("size") or 0.0) for w in words if w.get("size")]
    body_size = 0.0
    if sizes:
        sizes_sorted = sorted(sizes)
        body_size = sizes_sorted[len(sizes_sorted) // 2]  # median font size

    # Skip words fully inside any table bbox at word level
    def _word_inside_table(w: dict[str, Any]) -> bool:
        # Word bbox is [x0, top, x1, bottom]
        wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]
        for tx0, ty0, tx1, ty1 in table_bboxes:
            # Overlap/inside check. Midpoint check is highly robust
            mx = (wx0 + wx1) / 2.0
            my = (wy0 + wy1) / 2.0
            if tx0 - 1.0 <= mx <= tx1 + 1.0 and ty0 - 1.0 <= my <= ty1 + 1.0:
                return True
        return False

    filtered_words = [w for w in words if not _word_inside_table(w)]

    if not filtered_words:
        # All words are in tables, or no words. Return table elements.
        for el in elements:
            if "bbox_pts" in el:
                el["bbox"] = _to_pixels(el.pop("bbox_pts"))
        return {"page": page_num, "elements": elements}

    # Group words into horizontal lines
    lines: list[list[dict[str, Any]]] = []
    # Sort words by top then x0
    sorted_words = sorted(filtered_words, key=lambda w: (w["top"], w["x0"]))
    
    for w in sorted_words:
        placed = False
        for line in lines:
            line_top = min(lw["top"] for lw in line)
            line_bottom = max(lw["bottom"] for lw in line)
            # Vertical overlap check
            overlap = min(w["bottom"], line_bottom) - max(w["top"], line_top)
            h = w["bottom"] - w["top"]
            line_h = line_bottom - line_top
            min_h = min(h, line_h)
            if min_h > 0 and (overlap / min_h) >= 0.4:
                line.append(w)
                placed = True
                break
            elif abs(w["top"] - line_top) <= 2.5:  # baseline fallback
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # Sort words in each line horizontally
    for line in lines:
        line.sort(key=lambda w: w["x0"])

    # Sort the lines vertically by their top-most coordinate
    lines.sort(key=lambda line: min(w["top"] for w in line))

    # Horizontal split: divide each line into segments if there is a gap > column_split_threshold
    split_lines: list[list[dict[str, Any]]] = []
    column_split_threshold = max(20.0, body_size * 2.0)
    
    for line in lines:
        current_seg = [line[0]]
        for w in line[1:]:
            gap = w["x0"] - current_seg[-1]["x1"]
            if gap > column_split_threshold:
                split_lines.append(current_seg)
                current_seg = [w]
            else:
                current_seg.append(w)
        if current_seg:
            split_lines.append(current_seg)

    # Sort split lines vertically by their top coordinate
    split_lines.sort(key=lambda seg: min(w["top"] for w in seg))

    # Group segments into paragraphs/headings vertically
    # A block is a list of segments
    blocks: list[list[list[dict[str, Any]]]] = []
    line_gap_threshold = max(12.0, body_size * 1.8)

    for seg in split_lines:
        seg_top = min(w["top"] for w in seg)
        seg_bottom = max(w["bottom"] for w in seg)
        seg_x0 = min(w["x0"] for w in seg)
        seg_x1 = max(w["x1"] for w in seg)

        # Find the best block to merge into
        best_block = None
        best_gap = float("inf")

        for block in blocks:
            last_seg = block[-1]
            last_top = min(w["top"] for w in last_seg)
            last_bottom = max(w["bottom"] for w in last_seg)
            last_x0 = min(w["x0"] for w in last_seg)
            last_x1 = max(w["x1"] for w in last_seg)

            # Vertical proximity check
            v_gap = seg_top - last_bottom
            # A negative v_gap might happen with slight layout drifts; accept it up to -5.0
            if -5.0 <= v_gap <= line_gap_threshold:
                # Same column check (horizontal alignment or overlap)
                overlap_x = min(last_x1, seg_x1) - max(last_x0, seg_x0)
                w_last = last_x1 - last_x0
                w_seg = seg_x1 - seg_x0
                min_w = min(w_last, w_seg)

                same_col = False
                if min_w > 0 and (overlap_x / min_w) >= 0.4:
                    same_col = True
                elif abs(last_x0 - seg_x0) <= 20.0:  # left-aligned columns
                    same_col = True
                elif abs(last_x1 - seg_x1) <= 20.0:  # right-aligned columns
                    same_col = True

                if same_col:
                    if v_gap < best_gap:
                        best_gap = v_gap
                        best_block = block

        if best_block is not None:
            best_block.append(seg)
        else:
            blocks.append([seg])

    # Convert blocks to element dicts
    text_elements: list[dict[str, Any]] = []
    for para_lines in blocks:
        text_parts: list[str] = []
        x0s, y0s, x1s, y1s, ssizes = [], [], [], [], []
        for line in para_lines:
            text_parts.append(" ".join(w["text"] for w in line))
            x0s.append(min(w["x0"] for w in line))
            y0s.append(min(w["top"] for w in line))
            x1s.append(max(w["x1"] for w in line))
            y1s.append(max(w["bottom"] for w in line))
            ssizes.extend(float(w.get("size") or 0.0) for w in line)

        bbox = (min(x0s), min(y0s), max(x1s), max(y1s))
        avg_size = sum(ssizes) / len(ssizes) if ssizes else 0.0
        text = "\n".join(t.strip() for t in text_parts if t.strip()).strip()
        if not text:
            continue

        block_type = _classify_line(avg_size, body_size)
        if block_type == "heading":
            if len(para_lines) > 2 or len(text) > 150:
                block_type = "paragraph"

        text_elements.append(
            {
                "type": block_type,
                "text": text,
                "html": None,
                "bbox_pts": bbox,
                "language": None,
                "confidence": 1.0,
                "font_size": round(avg_size, 2) if avg_size else None,
            }
        )

    # Combine tables and text elements for global sorting
    all_elements = elements + text_elements

    # Use recursive XY-cut to sort globally in PDF points
    from ocr.layout_schema import LayoutBlock
    from ocr.reading_order import recursive_xy_cut

    # Map to LayoutBlock instances
    layout_blocks: list[LayoutBlock] = []
    for idx, el in enumerate(all_elements):
        bx0, by0, bx1, by1 = el["bbox_pts"]
        layout_blocks.append(
            LayoutBlock(
                block_id=f"temp_{idx}",
                type=el["type"],
                text=el["text"] or "",
                html=el["html"],
                x=bx0,
                y=by0,
                width=bx1 - bx0,
                height=by1 - by0,
                confidence=el["confidence"],
                font_size=el.get("font_size"),
            )
        )

    # Run recursive XY-cut in PDF points space
    sorted_layout_blocks = recursive_xy_cut(layout_blocks)

    # Convert back to sorted elements and scale coordinate to pixels
    elements_sorted: list[dict[str, Any]] = []
    for blk in sorted_layout_blocks:
        bbox_pts = (blk.x, blk.y, blk.x + blk.width, blk.y + blk.height)
        elements_sorted.append(
            {
                "type": blk.type,
                "text": blk.text,
                "html": blk.html,
                "bbox": _to_pixels(bbox_pts),
                "language": None,
                "confidence": blk.confidence,
                "font_size": blk.font_size,
            }
        )

    return {"page": page_num, "elements": elements_sorted}



def extract_native(pdf_path: str | Path) -> List[dict[str, Any]]:
    """Extract every page of a native PDF; return list of page dicts.

    Raises :class:`NativeExtractionError` if pdfplumber is unavailable or
    the PDF cannot be opened.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise NativeExtractionError(
            "pdfplumber is required for native PDF extraction. "
            "Install with: pip install pdfplumber"
        ) from exc

    path = Path(pdf_path)
    if not path.exists():
        raise NativeExtractionError(f"PDF not found: {path}")

    pages: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                pages.append(_extract_page(page, idx))
    except Exception as exc:
        raise NativeExtractionError(
            f"pdfplumber failed on {path.name}: {exc}"
        ) from exc

    logger.info(
        "native_extractor: %s → %d pages, %d elements",
        path.name,
        len(pages),
        sum(len(p["elements"]) for p in pages),
    )
    return pages


__all__ = ["extract_native", "NativeExtractionError", "PIXELS_PER_POINT"]
