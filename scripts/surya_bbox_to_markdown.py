"""
surya_bbox_to_markdown.py
=========================
Converts any Surya OCR result.json into clean, properly-aligned Markdown.

Key design goals
----------------
• Zero hardcoded pixel values  – every threshold is derived from the document's
  own statistics (median line height, page width, x-distribution of blocks).
• High-precision table detection – rows validated by vertical overlap, columns
  detected by horizontal range overlap. No arbitrary center gaps needed.
• Works on single-page and multi-page documents.
• Strips watermark / scanner noise lines automatically.
• Handles inline HTML tags already present in Surya text (<b>, <u>, etc.).

Usage
-----
    python surya_bbox_to_markdown.py result.json [output.md]

    # Process all result.json files inside a job directory
    python surya_bbox_to_markdown.py /path/to/job_dir/

If no output path is given the markdown is printed to stdout.
"""

import json
import sys
import os
import re
import statistics
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Tiny geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_x1(b): return b["bbox"][0]
def bbox_y1(b): return b["bbox"][1]
def bbox_x2(b): return b["bbox"][2]
def bbox_y2(b): return b["bbox"][3]
def bbox_cx(b): return (bbox_x1(b) + bbox_x2(b)) / 2
def bbox_cy(b): return (bbox_y1(b) + bbox_y2(b)) / 2
def bbox_h(b):  return bbox_y2(b) - bbox_y1(b)
def bbox_w(b):  return bbox_x2(b) - bbox_x1(b)


# ─────────────────────────────────────────────────────────────────────────────
# 1-D cluster helper
# ─────────────────────────────────────────────────────────────────────────────

def cluster_1d(values, gap):
    if not values:
        return []
    sv = sorted(values)
    clusters = [[sv[0]]]
    for v in sv[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(statistics.mean(c), c) for c in clusters]


# ─────────────────────────────────────────────────────────────────────────────
# Noise / watermark filter
# ─────────────────────────────────────────────────────────────────────────────

JUNK_PATTERNS = [
    re.compile(r'padasalai', re.IGNORECASE),
    re.compile(r'trb[-s]?tnpsc', re.IGNORECASE),
    re.compile(r'oken\s+scanner', re.IGNORECASE),
    re.compile(r'scanned\s+with', re.IGNORECASE),
    re.compile(r'kindly\s+send', re.IGNORECASE),
    re.compile(r'www\.', re.IGNORECASE),
    re.compile(r'\.com\s+of\s+\d', re.IGNORECASE),
    re.compile(r'\.net\s*[@\.]', re.IGNORECASE),
    re.compile(r'^[oO0]\s*$'),           # lone scanner artefact
]


def is_noise(block, page_h, page_w, median_h):
    text = block["text"].strip()
    if not text:
        return True
    h = bbox_h(block)
    if h < median_h * 0.3:
        return True
    if bbox_w(block) < page_w * 0.05 and len(text) <= 2:
        return True
    rel_y = bbox_y1(block) / page_h
    if rel_y < 0.03 or rel_y > 0.96:
        return True
    plain = re.sub(r'<[^>]+>', '', text)
    for pat in JUNK_PATTERNS:
        if pat.search(plain):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Layout classification
# ─────────────────────────────────────────────────────────────────────────────

def compute_page_stats(blocks):
    heights = [bbox_h(b) for b in blocks]
    x1s     = [bbox_x1(b) for b in blocks]
    x2s     = [bbox_x2(b) for b in blocks]
    med_h   = statistics.median(heights) if heights else 30
    med_x1  = statistics.median(x1s)     if x1s    else 0
    max_x2  = max(x2s) if x2s else 1000
    text_w  = max_x2 - med_x1
    return med_h, med_x1, text_w


def classify_indent(block, med_x1, text_w):
    delta = bbox_x1(block) - med_x1
    if delta > text_w * 0.10:
        return 'indent'
    return 'left'


def looks_like_section_label(block, med_h, med_x1, text_w):
    text  = re.sub(r'<[^>]+>', '', block["text"]).strip()
    short = bbox_w(block) < text_w * 0.45
    tall  = bbox_h(block) > med_h * 1.15
    colon = text.endswith(':')
    numbered = bool(re.match(r'^\d+[\.\)]', text))
    left  = classify_indent(block, med_x1, text_w) == 'left'
    return left and short and (colon or tall or numbered)


def looks_like_heading(block, med_h):
    return bbox_h(block) > med_h * 1.5


SEPARATOR_TEXTS = {'அல்லது', 'or', 'or:', '---', '–', '—', 'otherwise'}

def is_separator(block, med_h, text_w):
    text = clean_text(block["text"]).strip().lower()
    return text in SEPARATOR_TEXTS


# ─────────────────────────────────────────────────────────────────────────────
# Table detection & reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def detect_tables(blocks, med_h, page_w):
    if not blocks:
        return [], set()

    cy_values = [bbox_cy(b) for b in blocks]
    row_gap   = med_h * 1.8
    y_clusters = cluster_1d(cy_values, row_gap)

    cy_to_row = {}
    for centre, vals in y_clusters:
        for v in vals:
            cy_to_row[v] = centre

    block_row = {id(b): cy_to_row[bbox_cy(b)] for b in blocks}

    from collections import defaultdict
    rows = defaultdict(list)
    for b in blocks:
        rows[block_row[id(b)]].append(b)

    col_gap = page_w * 0.08
    table_row_centres = set()
    for row_centre, row_blocks in rows.items():
        if len(row_blocks) < 2:
            continue
        
        # Check if there is at least one pair of blocks that overlap vertically and are horizontally distinct
        has_overlap = False
        for i in range(len(row_blocks)):
            for j in range(i + 1, len(row_blocks)):
                b1 = row_blocks[i]
                b2 = row_blocks[j]
                y1_max = max(b1["bbox"][1], b2["bbox"][1])
                y2_min = min(b1["bbox"][3], b2["bbox"][3])
                overlap = y2_min - y1_max
                if overlap > 0:
                    min_h = min(bbox_h(b1), bbox_h(b2))
                    if overlap / min_h >= 0.3:
                        if abs(bbox_x1(b1) - bbox_x1(b2)) > col_gap:
                            has_overlap = True
                            break
            if has_overlap:
                break

        if has_overlap:
            table_row_centres.add(row_centre)

    if not table_row_centres:
        return [], set()

    sorted_row_centres = sorted(table_row_centres)
    table_groups = []
    current = [sorted_row_centres[0]]
    for rc in sorted_row_centres[1:]:
        if rc - current[-1] <= row_gap * 3:
            current.append(rc)
        else:
            table_groups.append(current)
            current = [rc]
    table_groups.append(current)

    # Filter out single-row tables (almost certainly false positives)
    valid_table_groups = [g for g in table_groups if len(g) >= 2]

    if not valid_table_groups:
        return [], set()

    table_block_ids = set()
    table_data = []
    for group in valid_table_groups:
        group_set = set(group)
        group_blocks = [b for b in blocks if block_row[id(b)] in group_set]
        for b in group_blocks:
            table_block_ids.add(id(b))
        
        # Horizontal overlap-based column detection!
        columns = []
        for b in sorted(group_blocks, key=lambda b: bbox_w(b), reverse=True):
            x1, x2 = bbox_x1(b), bbox_x2(b)
            merged = False
            for col in columns:
                overlap = min(x2, col[1]) - max(x1, col[0])
                if overlap > 0:
                    min_w = min(col[1] - col[0], x2 - x1)
                    if overlap / min_w >= 0.3:
                        col[0] = min(col[0], x1)
                        col[1] = max(col[1], x2)
                        merged = True
                        break
            if not merged:
                columns.append([x1, x2])
        
        columns.sort(key=lambda col: col[0])
        table_data.append((group, columns, rows))

    return table_data, table_block_ids


def assign_col(b, columns):
    x1, x2 = bbox_x1(b), bbox_x2(b)
    best_ci = 0
    best_overlap = -1
    for ci, col in enumerate(columns):
        overlap = min(x2, col[1]) - max(x1, col[0])
        if overlap > best_overlap:
            best_overlap = overlap
            best_ci = ci
    return best_ci


def render_table(group_row_centres, columns, rows_dict, block_row):
    n_cols = len(columns)
    sorted_row_centres = sorted(group_row_centres)
    all_row_data = []
    for rc in sorted_row_centres:
        row_blocks = rows_dict[rc]
        cells = [''] * n_cols
        for b in row_blocks:
            ci = assign_col(b, columns)
            txt = b["text"].strip()
            txt = re.sub(r'<b>(.*?)</b>', r'**\1**', txt)
            txt = re.sub(r'<u>(.*?)</u>', r'*\1*', txt)
            txt = re.sub(r'<[^>]+>', '', txt)
            cells[ci] = (cells[ci] + ' ' + txt).strip() if cells[ci] else txt
        all_row_data.append(cells)

    if not all_row_data:
        return ''

    # S.No autofill logic
    header = all_row_data[0]
    if header and header[0]:
        h_clean = re.sub(r'\s+', '', header[0].strip().lower())
        if h_clean in ('வ.எண்', 'வ.எண்.', 's.no', 's.no.', 'no', 'no.', 'sl.no', 'sl.no.', 'வ.எண்'):
            for idx in range(1, len(all_row_data)):
                if not all_row_data[idx][0].strip():
                    all_row_data[idx][0] = str(idx)

    lines  = []
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---'] * n_cols) + ' |')
    for row in all_row_data[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text)
    text = re.sub(r'<u>(.*?)</u>', r'*\1*', text)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*',  text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main page → markdown converter
# ─────────────────────────────────────────────────────────────────────────────

def page_to_markdown(page):
    all_blocks = page["blocks"]
    page_w = page.get("width", 3000)
    page_h = page.get("height", 4000)

    if not all_blocks:
        return ''

    has_reading_order = all("reading_order" in b for b in all_blocks)
    if has_reading_order:
        all_blocks = sorted(all_blocks, key=lambda b: b["reading_order"])
    else:
        all_blocks = sorted(all_blocks, key=lambda b: (bbox_y1(b), bbox_x1(b)))

    med_h, med_x1, text_w = compute_page_stats(all_blocks)
    clean_blocks = [b for b in all_blocks if not is_noise(b, page_h, page_w, med_h)]

    if not clean_blocks:
        return ''

    med_h, med_x1, text_w = compute_page_stats(clean_blocks)

    layout_blocks = []
    table_candidates = []

    for b in clean_blocks:
        is_h = looks_like_heading(b, med_h)
        is_l = looks_like_section_label(b, med_h, med_x1, text_w)
        is_s = is_separator(b, med_h, text_w)
        
        if is_h or is_l or is_s:
            layout_blocks.append(b)
        else:
            table_candidates.append(b)

    table_data, table_block_ids = detect_tables(table_candidates, med_h, page_w)

    emitted_tables = set()
    pending_para   = []
    pending_type   = None
    prev_y2        = None

    # Map block id → table group index
    block_to_table_group = {}
    for gi, (group_row_centres, col_centres, rows_dict) in enumerate(table_data):
        group_set   = set(group_row_centres)
        cy_values  = [bbox_cy(b) for b in table_candidates]
        row_gap    = med_h * 1.8
        y_clusters = cluster_1d(cy_values, row_gap)
        cy_to_row  = {}
        for centre, vals in y_clusters:
            for v in vals:
                cy_to_row[v] = centre
        for b in table_candidates:
            row_centre = cy_to_row.get(bbox_cy(b))
            if row_centre in group_set and id(b) in table_block_ids:
                block_to_table_group[id(b)] = gi

    def flush_para():
        nonlocal pending_para, pending_type
        if not pending_para:
            return None
        text = ' '.join(pending_para)
        t    = pending_type
        pending_para = []
        pending_type = None
        return (t, text)

    result_items = []
    first_heading_seen = False

    for b in sorted(clean_blocks, key=lambda b: (bbox_y1(b), bbox_x1(b))):
        y1 = bbox_y1(b)

        if id(b) in block_to_table_group:
            gi = block_to_table_group[id(b)]
            if gi not in emitted_tables:
                para = flush_para()
                if para:
                    result_items.append((prev_y2 or y1, para[0], para[1]))
                
                g_row_centres, col_centres, rows_dict = table_data[gi]
                cy_values  = [bbox_cy(bl) for bl in table_candidates]
                row_gap    = med_h * 1.8
                y_clusters = cluster_1d(cy_values, row_gap)
                cy_to_row  = {}
                for centre, vals in y_clusters:
                    for v in vals:
                        cy_to_row[v] = centre
                block_row_map = {id(bl): cy_to_row[bbox_cy(bl)] for bl in table_candidates}
                md = render_table(g_row_centres, col_centres, rows_dict, block_row_map)
                result_items.append((y1, 'table', md))
                emitted_tables.add(gi)
            prev_y2 = bbox_y2(b)
            continue

        text = clean_text(b["text"])
        if not text:
            continue

        is_heading = looks_like_heading(b, med_h)
        is_label   = looks_like_section_label(b, med_h, med_x1, text_w)
        is_sep     = is_separator(b, med_h, text_w)
        indent     = classify_indent(b, med_x1, text_w)
        btype      = ('heading'      if is_heading
                      else 'label'   if is_label
                      else 'sep'     if is_sep
                      else 'indent_body' if indent == 'indent'
                      else 'body')

        y_gap = (y1 - prev_y2) if prev_y2 is not None else 0
        large_gap = y_gap > med_h * 1.8

        if btype in ('heading', 'label', 'sep') or large_gap or (pending_type and btype != pending_type):
            para = flush_para()
            if para:
                result_items.append((prev_y2 or y1, para[0], para[1]))

        if btype == 'sep':
            result_items.append((y1, 'sep', text))
        elif btype == 'heading':
            if not first_heading_seen:
                result_items.append((y1, 'heading_1', text))
                first_heading_seen = True
            else:
                result_items.append((y1, 'heading_2', text))
        else:
            if pending_type is None:
                pending_type = btype
            pending_para.append(text)

        prev_y2 = bbox_y2(b)

    para = flush_para()
    if para:
        result_items.append((prev_y2 or 0, para[0], para[1]))

    lines = []
    for (_, rtype, content) in result_items:
        if rtype == 'table':
            lines.append('\n' + content + '\n')
        elif rtype == 'heading_1':
            lines.append('\n# ' + content + '\n')
        elif rtype == 'heading_2':
            lines.append('\n## ' + content + '\n')
        elif rtype == 'label':
            lines.append('\n**' + content + '**')
        elif rtype == 'sep':
            lines.append('\n---\n\n*' + content + '*\n')
        elif rtype == 'indent_body':
            lines.append(content)
        else:
            lines.append(content)

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def convert_file(json_path: Path, output_path: Path | None = None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pages = data.get("pages", [])
    md_parts = []

    for page in pages:
        pg_md = page_to_markdown(page)
        if pg_md.strip():
            md_parts.append(pg_md)

    markdown = '\n\n---\n\n'.join(md_parts)

    if output_path:
        output_path.write_text(markdown, encoding='utf-8')
        try:
            print(f"Saved -> {output_path}")
        except UnicodeEncodeError:
            pass
    else:
        try:
            print(markdown)
        except UnicodeEncodeError:
            try:
                sys.stdout.buffer.write(markdown.encode(sys.stdout.encoding or 'utf-8', errors='replace'))
                print()
            except Exception:
                pass

    return markdown


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = Path(sys.argv[1])

    if arg.is_dir():
        json_files = list(arg.rglob("result.json"))
        if not json_files:
            print(f"No result.json found under {arg}")
            sys.exit(1)
        for jf in json_files:
            out = jf.parent / "result.md"
            convert_file(jf, out)
    elif arg.is_file():
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        convert_file(arg, out)
    else:
        print(f"Path not found: {arg}")
        sys.exit(1)


if __name__ == "__main__":
    main()