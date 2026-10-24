"""Excel document worker.

Reads one sheet from an ``.xlsx`` file, converts every row into a
structured text block, and stores the result through the standard
``FileHandler`` / chunking pipeline so the AI chat can query it.

Text format per row
-------------------
Each non-empty row is rendered as a block that begins with the
section-number column (first column whose header contains "section" or
"s.no" or "sl" — detected automatically, or column A as fallback),
followed by every other column as ``Key: Value`` lines.  Empty rows and
rows whose every cell is blank are skipped.

Example output for a Stamp Act sheet::

    === Section 3 ===
    Description: Instruments chargeable with duty.
    Article: 1
    Proper Stamp Duty: Rs. 5/-

This aligned format allows the LLM to answer questions like
"what is the stamp duty for section 3?" accurately.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_section_col(headers: list[str]) -> int:
    """Return the column index most likely to be the section/serial number."""
    keywords = ("section", "sec", "s.no", "sl.no", "sl", "sno", "no", "number")
    for i, h in enumerate(headers):
        hl = str(h).lower().strip()
        for kw in keywords:
            if kw in hl:
                return i
    return 0  # fall back to first column


def _sheet_to_text(df) -> str:  # df: pd.DataFrame
    """Convert a DataFrame into section-aligned plain text with row/col context for LLMs."""
    headers = [str(c).strip() for c in df.columns]
    
    def get_col_name(idx, header):
        if header.startswith("Unnamed:") or not header:
            # Convert 0-indexed column to Excel letter (A, B, C...)
            col_letter = ""
            temp = idx
            while temp >= 0:
                col_letter = chr(temp % 26 + 65) + col_letter
                temp = temp // 26 - 1
            return f"Column {col_letter}"
        return header

    clean_headers = [get_col_name(i, h) for i, h in enumerate(headers)]
    section_col = _detect_section_col(headers)
    other_cols = [i for i in range(len(headers)) if i != section_col]

    blocks: list[str] = []
    for idx, row in df.iterrows():
        excel_row = idx + 2  # Assuming row 1 is the header
        
        values = [str(v).strip() if str(v).strip() not in ("", "nan", "None") else "" for v in row]
        if all(v == "" for v in values):
            continue  # blank row

        sec_val = values[section_col]
        section_title = f"Section {sec_val}" if sec_val else "Entry"

        lines = [f"=== Row {excel_row} | {section_title} ==="]
        for i in other_cols:
            val = values[i]
            if val:
                lines.append(f"{clean_headers[i]}: {val}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def list_sheets(xlsx_path: Path) -> list[str]:
    """Return the sheet names in the workbook without loading data."""
    import openpyxl
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    names = wb.sheetnames
    wb.close()
    return names


# ── worker ────────────────────────────────────────────────────────────────────

class ExcelWorker:
    """Process one sheet of an Excel file end-to-end (extract → store → index)."""

    def __init__(
        self,
        job_id: str,
        xlsx_path: Path,
        sheet_name: str,
        storage,
        progress_tracker,
        resume_handler,
        job_manager=None,
    ):
        self.job_id = job_id
        self.xlsx_path = xlsx_path
        self.sheet_name = sheet_name
        self.storage = storage
        self.progress_tracker = progress_tracker
        self.resume_handler = resume_handler
        self.job_manager = job_manager  # for scheduling indexing after extraction

    # ── public entry point ─────────────────────────────────────────────────

    def process(self) -> None:
        """Full pipeline: extract → store → trigger indexing."""
        logger.info(
            "ExcelWorker: job=%s file=%s sheet=%r",
            self.job_id, self.xlsx_path.name, self.sheet_name,
        )
        try:
            self.progress_tracker.start_job(self.job_id)
            full_text, html_table = self._extract_sheet()
            self._store(full_text, html_table)
            self._build_outputs()
            logger.info("ExcelWorker: extraction complete for job %s", self.job_id)

            # Schedule semantic indexing on the main event-loop (same as PDF path)
            if self.job_manager is not None and self.job_manager.loop is not None:
                self.job_manager.schedule_indexing(self.job_id, force=True)
                logger.info(
                    "ExcelWorker: scheduling indexing for job %s on main event loop",
                    self.job_id,
                )

        except Exception as exc:
            msg = f"Excel processing failed: {exc}"
            logger.error(msg, exc_info=True)
            self.progress_tracker.mark_failed(self.job_id, msg)
            raise

    # ── internals ─────────────────────────────────────────────────────────

    def _extract_sheet(self) -> tuple[str, str]:
        import pandas as pd

        available = list_sheets(self.xlsx_path)
        if self.sheet_name not in available:
            # Fall back to first sheet
            logger.warning(
                "Sheet %r not found in %s — falling back to first sheet %r",
                self.sheet_name, self.xlsx_path.name, available[0],
            )
            self.sheet_name = available[0]

        df = pd.read_excel(str(self.xlsx_path), sheet_name=self.sheet_name, dtype=str)
        df.dropna(how="all", inplace=True)   # drop completely empty rows
        df.fillna("", inplace=True)

        text = _sheet_to_text(df)
        
        # Generate a clean HTML table for the preview
        html_table = df.to_html(index=False, classes="table-spreadsheet", border=0)
        
        logger.info(
            "ExcelWorker: extracted %d chars and HTML table from sheet %r (%d rows)",
            len(text), self.sheet_name, len(df),
        )
        return text, html_table

    def _store(self, text: str, html: str) -> None:
        """Write the extracted text and structured JSON."""
        self.storage.append_text(self.job_id, text, page_num=1)
        
        page_dict = {
            "page": 1,
            "elements": [
                {
                    "type": "table",
                    "text": text,
                    "html": html
                }
            ]
        }
        self.storage.save_page_structure(self.job_id, 1, page_dict)
        
        self.progress_tracker.update_progress(self.job_id, 1, success=True)
        logger.info("ExcelWorker: stored text and structure for job %s", self.job_id)

    def _build_outputs(self) -> None:
        """Build result.json / result.html.  Progress is already marked complete
        by :meth:`update_progress` once all pages are accounted for."""
        if hasattr(self.storage, "build_combined_result"):
            self.storage.build_combined_result(self.job_id)
        if hasattr(self.storage, "build_result_html"):
            self.storage.build_result_html(self.job_id)
        logger.info("ExcelWorker: built outputs for job %s", self.job_id)
