import os
import json
import threading
import shutil
import uuid
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations for job storage"""

    # Per-job locks so concurrent readers/writers never corrupt status.json
    _status_locks: Dict[str, threading.Lock] = {}
    _locks_meta_lock = threading.Lock()  # protects the dict itself

    def _get_status_lock(self, job_id: str) -> threading.Lock:
        with self._locks_meta_lock:
            if job_id not in self._status_locks:
                self._status_locks[job_id] = threading.Lock()
            return self._status_locks[job_id]
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_job_directory(self, job_id: str) -> Path:
        """
        Create a directory for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Path to job directory
        """
        job_dir = self.output_dir / job_id
        job_dir.mkdir(exist_ok=True)
        # Create pages sub-directory for structured JSON
        (job_dir / "pages").mkdir(exist_ok=True)
        return job_dir

    @contextmanager
    def processing_lock(self):
        """Cross-process lock enforcing one active document worker per output store."""
        lock_path = self.output_dir / ".processing.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+b") as lock_file:
            if lock_file.seek(0, os.SEEK_END) == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def get_job_directory(self, job_id: str) -> Path:
        """Get path to job directory"""
        return self.output_dir / job_id
    
    def get_result_file_path(self, job_id: str) -> Path:
        """Get path to result text file"""
        return self.get_job_directory(job_id) / "result.txt"
    
    def get_status_file_path(self, job_id: str) -> Path:
        """Get path to status JSON file"""
        return self.get_job_directory(job_id) / "status.json"
    
    def get_error_log_path(self, job_id: str) -> Path:
        """Get path to error log file"""
        return self.get_job_directory(job_id) / "error.log"

    def get_page_json_path(self, job_id: str, page_num: int) -> Path:
        """Get path to a structured page JSON file."""
        return self.get_job_directory(job_id) / "pages" / f"page_{page_num:04d}.json"

    def get_structured_result_path(self, job_id: str) -> Path:
        """Get path to the combined structured JSON result."""
        return self.get_job_directory(job_id) / "result.json"

    def get_html_result_path(self, job_id: str) -> Path:
        """Get path to the HTML result."""
        return self.get_job_directory(job_id) / "result.html"

    def get_source_pdf_path(self, job_id: str) -> Path:
        """Get path to the job-scoped source PDF copy."""
        return self.get_job_directory(job_id) / "source.pdf"

    def get_source_file_path(self, job_id: str, extension: str) -> Path:
        """Return the immutable, job-scoped original source path."""
        suffix = extension.lower().lstrip(".")
        if suffix not in {"pdf", "docx", "xlsx"}:
            raise ValueError(f"Unsupported source extension: {extension}")
        return self.get_job_directory(job_id) / f"source.{suffix}"

    def save_source_file(self, job_id: str, source_path: Path) -> Path:
        """Atomically persist an original source inside its job directory."""
        target_path = self.get_source_file_path(job_id, source_path.suffix)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(source_path, temp_path)
            with temp_path.open("rb+") as source:
                source.flush()
                os.fsync(source.fileno())
            os.replace(temp_path, target_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return target_path

    def save_source_pdf(self, job_id: str, pdf_path: Path) -> Path:
        """Persist a copy of the source PDF alongside job outputs."""
        return self.save_source_file(job_id, pdf_path)

    # ------------------------------------------------------------------
    # Plain text (backward compatible — live append)
    # ------------------------------------------------------------------

    def append_text(self, job_id: str, text: str, page_num: int) -> None:
        """
        Persist one page idempotently and atomically rebuild the combined text.
        
        Args:
            job_id: Job identifier
            text: Text to append
            page_num: Page number
        """
        result_file = self.get_result_file_path(job_id)
        pages_dir = self.get_job_directory(job_id) / "pages"
        pages_dir.mkdir(exist_ok=True)
        page_file = pages_dir / f"page_{page_num:04d}.txt"
        try:
            self._migrate_legacy_result_pages(job_id)
            page_temp = page_file.with_name(f".{page_file.name}.{uuid.uuid4().hex}.tmp")
            with open(page_temp, 'w', encoding='utf-8') as f:
                f.write(text)
                f.flush()  # Immediate flush to disk
                os.fsync(f.fileno())  # Force write to disk
            os.replace(page_temp, page_file)

            result_temp = result_file.with_name(f".{result_file.name}.{uuid.uuid4().hex}.tmp")
            with open(result_temp, 'w', encoding='utf-8') as combined:
                for index, stored_page in enumerate(sorted(pages_dir.glob("page_*.txt"))):
                    match = re.search(r"page_(\d+)\.txt$", stored_page.name)
                    if not match:
                        continue
                    if index:
                        combined.write("\n\n")
                    combined.write(f"--- Page {int(match.group(1))} ---\n")
                    combined.write(stored_page.read_text(encoding="utf-8"))
                combined.flush()
                os.fsync(combined.fileno())
            os.replace(result_temp, result_file)
        except Exception as e:
            logger.error(f"Failed to append text for job {job_id}, page {page_num}: {e}")
            self.log_error(job_id, f"Page {page_num}: Failed to write text - {str(e)}")
        finally:
            if 'page_temp' in locals():
                page_temp.unlink(missing_ok=True)
            if 'result_temp' in locals():
                result_temp.unlink(missing_ok=True)

    def _migrate_legacy_result_pages(self, job_id: str) -> None:
        """Seed page text artifacts from an older append-only result once."""
        pages_dir = self.get_job_directory(job_id) / "pages"
        result_file = self.get_result_file_path(job_id)
        if not result_file.exists() or result_file.stat().st_size == 0:
            return
        content = result_file.read_text(encoding="utf-8")
        matches = list(re.finditer(r"(?:^|\n\n)--- Page (\d+) ---\n", content))
        for index, match in enumerate(matches):
            page_num = int(match.group(1))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            target = pages_dir / f"page_{page_num:04d}.txt"
            if not target.exists():
                target.write_text(content[start:end].rstrip("\n"), encoding="utf-8")

    # ------------------------------------------------------------------
    # Structured page storage (new — per-page JSON)
    # ------------------------------------------------------------------

    def save_page_structure(self, job_id: str, page_num: int, page_dict: dict) -> None:
        """
        Save structured page result as JSON.

        Args:
            job_id: Job identifier
            page_num: Page number
            page_dict: dict with keys 'page', 'elements' (from PageResult.to_dict())
        """
        pages_dir = self.get_job_directory(job_id) / "pages"
        pages_dir.mkdir(exist_ok=True)
        page_file = pages_dir / f"page_{page_num:04d}.json"

        temp_file = page_file.with_name(f".{page_file.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(page_dict, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, page_file)
        except Exception as e:
            logger.error(f"Failed to save page structure for job {job_id}, page {page_num}: {e}")
        finally:
            temp_file.unlink(missing_ok=True)

    def load_page_structure(self, job_id: str, page_num: int) -> Optional[dict]:
        """Load a single page's structured JSON."""
        page_file = self.get_page_json_path(job_id, page_num)
        if not page_file.exists():
            return None
        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load page structure for job {job_id}, page {page_num}: {e}")
            return None

    def build_combined_structured_result(self, job_id: str) -> Optional[dict]:
        """
        Combine all per-page JSONs into one document-level result.json.
        Called once after processing completes.

        For Sur OCRpages, output matches pipeline_v2 format:
        ``{"pages": [{page, image, width, height, blocks: [...]}]}``

        Returns the combined dict or None on failure.
        """
        pages_dir = self.get_job_directory(job_id) / "pages"
        if not pages_dir.exists():
            return None

        page_files = sorted(pages_dir.glob("page_*.json"))
        if not page_files:
            return None

        pages: List[dict] = []
        is_surya = False
        for pf in page_files:
            try:
                with open(pf, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                # Detect Surya pages by presence of extended block fields.
                elements = page_data.get("elements", [])
                if elements and elements[0].get("reading_order") is not None and elements[0].get("bbox"):
                    is_surya = True
                pages.append(page_data)
            except Exception:
                continue

        if is_surya:
            # Convert to pipeline_v2 format: elements -> blocks
            surya_pages = []
            for p in pages:
                blocks = []
                for el in p.get("elements", []):
                    blocks.append({
                        "id": el.get("id", ""),
                        "text": el.get("text", ""),
                        "bbox": el.get("bbox", []),
                        "confidence": el.get("confidence", 0),
                        "reading_order": el.get("reading_order", 0),
                        "cleaned": el.get("cleaned", False),
                    })
                surya_page: dict = {"page": p["page"], "blocks": blocks}
                if p.get("image"):
                    surya_page["image"] = p["image"]
                if p.get("width"):
                    surya_page["width"] = p["width"]
                if p.get("height"):
                    surya_page["height"] = p["height"]
                surya_pages.append(surya_page)
            combined = {"pages": surya_pages}
        else:
            combined = {"total_pages": len(pages), "pages": pages}

        # Write combined result.json
        out_path = self.get_structured_result_path(job_id)
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"Built combined result.json for job {job_id} ({len(pages)} pages)")
        except Exception as e:
            logger.error(f"Failed to write combined result.json for job {job_id}: {e}")

        return combined

    def build_html_result(self, job_id: str) -> Optional[str]:
        """
        Generate result.html from per-page JSONs.
        Preserves table HTML, wraps other elements in appropriate tags.

        Returns the HTML string or None.
        """
        pages_dir = self.get_job_directory(job_id) / "pages"
        if not pages_dir.exists():
            return None

        page_files = sorted(pages_dir.glob("page_*.json"))
        if not page_files:
            return None

        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head><meta charset=\"utf-8\"><title>OCR Result</title>",
            "<style>",
            "body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#333;background:#fafafa}",
            ".page-container{background:#fff;padding:2.5rem;margin-bottom:2rem;box-shadow:0 4px 6px rgba(0,0,0,0.05);border-radius:8px}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0}",
            "td,th{border:1px solid #ccc;padding:6px 10px;text-align:left}",
            "th{background:#f5f5f5}",
            ".page-break{border-top:2px dashed #999;margin:2rem 0;padding-top:0.5rem;"
            "color:#666;font-size:0.85rem;font-weight:bold}",
            "h2,h3{margin-top:1.5rem;color:#111;text-align:center}",
            "p{margin:0.5rem 0}",
            ".docx-title{font-size:1.75rem;font-weight:700;text-align:center;margin:0 0 1.25rem}",
            ".docx-heading{font-weight:700;margin:1.25rem 0 0.5rem;color:#111}",
            ".docx-paragraph{margin:0.45rem 0}",
            ".docx-list{margin:0.35rem 0 0.35rem 1.25rem}",
            ".docx-table{border-collapse:collapse;width:100%;margin:1rem 0;table-layout:auto}",
            ".docx-table th{background:#f3f4f6;font-weight:700}",
            "/* Layout tables styling */",
            ".layout-table{border-collapse:collapse;width:100%;margin:0.2rem 0;table-layout:fixed}",
            ".layout-table td{border:none !important;padding:4px 8px !important;vertical-align:top;margin:0 !important}",
            ".layout-cell{word-wrap:break-word;overflow-wrap:break-word}",
            ".layout-gap{padding:0 !important;margin:0 !important}",
            "</style></head><body>",
        ]

        for pf in page_files:
            try:
                with open(pf, 'r', encoding='utf-8') as f:
                    page = json.load(f)
            except Exception:
                continue

            page_num = page.get("page", "?")
            html_parts.append(f'<div class="page-container">')
            html_parts.append(f'<div id="ocr-page-{page_num}" class="page-break">Page {page_num}</div>')

            W = page.get("width")
            H = page.get("height")
            elements = page.get("elements", [])

            if not W or not H:
                all_x2s = [el["bbox"][2] for el in elements if el.get("bbox") and len(el["bbox"]) >= 4]
                all_y2s = [el["bbox"][3] for el in elements if el.get("bbox") and len(el["bbox"]) >= 4]
                W = max(all_x2s) if all_x2s else 2000
                H = max(all_y2s) if all_y2s else 3000

            # 1. Calculate dynamic vertical overlap tolerance
            valid_heights = []
            for el in elements:
                bbox = el.get("bbox")
                if bbox and len(bbox) >= 4:
                    valid_heights.append(bbox[3] - bbox[1])

            avg_height = sum(valid_heights) / len(valid_heights) if valid_heights else 20.0
            tolerance = max(15.0, min(avg_height * 0.45, 40.0))

            # 2. Separate visual and non-visual elements
            visual_elements = []
            non_visual_elements = []

            for el in elements:
                etype = el.get("type", "paragraph")
                bbox = el.get("bbox")
                if etype == "table" or not bbox or len(bbox) < 4:
                    non_visual_elements.append(el)
                else:
                    visual_elements.append(el)

            # Sort visual elements by reading_order, y1, x1
            visual_elements.sort(key=lambda x: (x.get("reading_order", 0), x["bbox"][1], x["bbox"][0]))

            # Perform vertical clustering into lines/rows
            rows = []
            for el in visual_elements:
                inserted = False
                el_x1, el_y1, el_x2, el_y2 = el["bbox"]
                el_h = el_y2 - el_y1

                if rows:
                    last_row = rows[-1]
                    has_horiz_overlap = False
                    for item in last_row:
                        item_x1, item_y1, item_x2, item_y2 = item["bbox"]
                        overlap_x = max(0.0, min(item_x2, el_x2) - max(item_x1, el_x1))
                        min_w = min(item_x2 - item_x1, el_x2 - el_x1)
                        horiz_overlap_ratio = overlap_x / min_w if min_w > 0 else 0.0
                        if horiz_overlap_ratio > 0.1:  # More than 10% horizontal overlap
                            has_horiz_overlap = True
                            break

                    if not has_horiz_overlap:
                        row_y1s = [item["bbox"][1] for item in last_row]
                        row_y2s = [item["bbox"][3] for item in last_row]

                        row_min_y1 = min(row_y1s)
                        row_max_y2 = max(row_y2s)
                        row_h = row_max_y2 - row_min_y1

                        overlap_y1 = max(row_min_y1, el_y1)
                        overlap_y2 = min(row_max_y2, el_y2)
                        overlap = max(0.0, overlap_y2 - overlap_y1)

                        min_h = min(row_h, el_h)
                        overlap_ratio = overlap / min_h if min_h > 0 else 0.0

                        center_diff = abs((row_min_y1 + row_max_y2)/2.0 - (el_y1 + el_y2)/2.0)

                        if overlap_ratio > 0.3 or center_diff < tolerance:
                            last_row.append(el)
                            inserted = True

                if not inserted:
                    rows.append([el])

            # Interleave visual and non-visual elements by vertical position
            all_renderable_rows = []
            for el in non_visual_elements:
                bbox = el.get("bbox")
                y_val = bbox[1] if (bbox and len(bbox) >= 2) else 999999
                all_renderable_rows.append((y_val, False, el))

            for row in rows:
                row_y1s = [item["bbox"][1] for item in row]
                avg_y1 = sum(row_y1s) / len(row_y1s) if row_y1s else 0
                all_renderable_rows.append((avg_y1, True, row))

            # Sort everything by vertical position
            all_renderable_rows.sort(key=lambda x: x[0])

            for _, is_visual_row, item in all_renderable_rows:
                if not is_visual_row:
                    el = item
                    etype = el.get("type", "paragraph")
                    text = el.get("text", "")
                    html_val = el.get("html", "")

                    if etype == "table" and html_val:
                        html_parts.append(html_val)
                    elif html_val:
                        html_parts.append(html_val)
                    elif etype == "figure":
                        html_parts.append('<p style="text-align:center;"><em>[figure]</em></p>')
                    else:
                        html_parts.append(f"<p>{_esc(text)}</p>")
                else:
                    row_elements = item
                    row_elements.sort(key=lambda x: x["bbox"][0])

                    if len(row_elements) == 1:
                        el = row_elements[0]
                        text = el.get("text", "")
                        etype = el.get("type", "paragraph")
                        esc_text = _esc(text)

                        # Heading detection with refined triggers
                        is_heading = False
                        text_strip = text.strip()
                        text_upper = text_strip.upper()
                        if etype == "title" or (len(text_strip) < 100 and (
                            text_upper.startswith("SECTION ") or
                            text_upper.startswith("CHAPTER ") or
                            text_upper == "SECTION" or
                            text_upper == "CHAPTER" or
                            text_upper.startswith("PREFACE") or
                            text_upper.startswith("GENERAL CONTENTS") or
                            text_strip.isupper()
                        )):
                            is_heading = True

                        if is_heading:
                            html_parts.append(f"<h2>{esc_text}</h2>")
                        elif etype == "list":
                            items = text.split("\n")
                            html_parts.append("<ul>")
                            for it in items:
                                if it.strip():
                                    html_parts.append(f"  <li>{_esc(it.strip())}</li>")
                            html_parts.append("</ul>")
                        else:
                            html_parts.append(f"<p>{esc_text}</p>")
                    else:
                        # Multi-column grid layout reconstruction
                        html_parts.append('<table class="layout-table"><tr>')

                        col_widths = []
                        current_x = 0
                        for el in row_elements:
                            el_x1, el_y1, el_x2, el_y2 = el["bbox"]
                            if el_x1 > current_x:
                                col_widths.append((True, None, el_x1 - current_x))
                            col_widths.append((False, el, el_x2 - el_x1))
                            current_x = el_x2

                        if W > current_x:
                            col_widths.append((True, None, W - current_x))

                        total_pixels = sum(w for _, _, w in col_widths)

                        for is_gap, cell_el, w in col_widths:
                            pct = (w / total_pixels) * 100.0 if total_pixels > 0 else 0
                            if is_gap:
                                html_parts.append(f'<td class="layout-gap" style="width: {pct:.2f}%;"></td>')
                            else:
                                text = cell_el.get("text", "")
                                etype = cell_el.get("type", "paragraph")
                                esc_text = _esc(text)

                                if etype == "list" or "\n" in text:
                                    items = text.split("\n")
                                    cell_content = "<ul>" + "".join(f"<li>{_esc(it.strip())}</li>" for it in items if it.strip()) + "</ul>"
                                else:
                                    cell_content = esc_text

                                html_parts.append(f'<td class="layout-cell" style="width: {pct:.2f}%;">{cell_content}</td>')

                        html_parts.append('</tr></table>')

            html_parts.append(f"</div><!-- page-container -->")

        script_str = """
<script>
window.addEventListener("scroll", function() {
    let closestPage = "1";
    let markers = document.querySelectorAll('[id^="ocr-page-"]');
    for (let i = 0; i < markers.length; i++) {
        let rect = markers[i].getBoundingClientRect();
        if (rect.top <= 100) {
            closestPage = markers[i].id.replace("ocr-page-", "");
        } else {
            break;
        }
    }
    window.parent.postMessage({ type: "ocr_scroll", page: closestPage }, "*");
});
</script>
"""
        html_parts.append(script_str + "</body></html>")
        html_str = "\n".join(html_parts)

        out_path = self.get_html_result_path(job_id)
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html_str)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"Built result.html for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to write result.html for job {job_id}: {e}")

        return html_str

    def read_structured_result(self, job_id: str) -> Optional[dict]:
        """Read the combined result.json."""
        path = self.get_structured_result_path(job_id)
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read structured result for job {job_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Status / error / existing methods (unchanged)
    # ------------------------------------------------------------------
    
    def save_status(self, job_id: str, status_data: Dict[str, Any]) -> None:
        """
        Save job status to JSON file.
        Uses a per-job lock plus temp-file/fsync/replace so concurrent readers
        and process restarts never observe a partially written status document.

        Args:
            job_id: Job identifier
            status_data: Status dictionary
        """
        status_file = self.get_status_file_path(job_id)

        try:
            status_data['last_updated'] = datetime.now().isoformat()

            with self._get_status_lock(job_id):
                temp_file = status_file.with_name(f".{status_file.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(status_data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_file, status_file)
                finally:
                    temp_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to save status for job {job_id}: {e}")
    
    def load_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load job status from JSON file.
        Holds the per-job lock while reading so we never read a mid-write truncated file.

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary or None if not found
        """
        status_file = self.get_status_file_path(job_id)

        if not status_file.exists():
            return None

        try:
            with self._get_status_lock(job_id):
                with open(status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load status for job {job_id}: {e}")
            return None
    
    def log_error(self, job_id: str, error_message: str) -> None:
        """
        Log error to error log file
        
        Args:
            job_id: Job identifier
            error_message: Error message
        """
        error_log = self.get_error_log_path(job_id)
        
        try:
            with open(error_log, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] {error_message}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Failed to log error for job {job_id}: {e}")
    
    def read_result(self, job_id: str) -> Optional[str]:
        """
        Read result text file
        
        Args:
            job_id: Job identifier
            
        Returns:
            Result text or None if not found
        """
        result_file = self.get_result_file_path(job_id)
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read result for job {job_id}: {e}")
            return None
    
    def job_exists(self, job_id: str) -> bool:
        """Check if job directory exists"""
        return self.get_job_directory(job_id).exists()
    
    def download_source_pdf(self, job_id: str, target_path: Path) -> bool:
        """Recover the source PDF from the job-scoped local copy if present."""
        if target_path.exists():
            return True

        source_path = self.get_source_pdf_path(job_id)
        if not source_path.exists():
            return False

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return True

    def list_jobs(self) -> List[str]:
        """List all job IDs in local outputs directory"""
        if not self.output_dir.exists():
            return []
        
        job_ids = []
        for entry in self.output_dir.iterdir():
            if entry.is_dir() and (entry / "status.json").exists():
                job_ids.append(entry.name)
        return job_ids

    def cleanup_job(self, job_id: str) -> None:
        """
        Delete job directory and all its contents

        Args:
            job_id: Job identifier
        """
        job_dir = self.get_job_directory(job_id)

        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up job directory: {job_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup job {job_id}: {e}")

        # Release the per-job lock so it can be GC'd
        with self._locks_meta_lock:
            self._status_locks.pop(job_id, None)



def _esc(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
