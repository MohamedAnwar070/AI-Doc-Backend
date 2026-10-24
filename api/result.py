from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import logging
import asyncio

from core.job_manager import JobManager, get_job_manager
from core.source_preview import SourcePreviewError, create_office_pdf_preview
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["result"])


def _looks_like_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(5).startswith(b"%PDF-")
    except OSError:
        return False


def _store_source_preview_status(job_manager: JobManager, job_id: str, pages: int, error: str = "") -> None:
    status_info = job_manager.progress_tracker.get_status(job_id)
    if status_info.get("status") == "not_found":
        return
    status_info["source_total_pages"] = pages
    status_info["source_preview_error"] = error
    job_manager.storage.save_status(job_id, status_info)


def _repair_office_source_preview(job_manager: JobManager, job_id: str, pdf_filename: str, source_path: Path) -> tuple[bool, str]:
    upload_path = Path(settings.upload_dir) / pdf_filename
    if not upload_path.exists():
        return False, f"Uploaded source file for {pdf_filename} was not found"

    try:
        _, pages = create_office_pdf_preview(upload_path, source_path)
    except SourcePreviewError as exc:
        detail = str(exc)
        _store_source_preview_status(job_manager, job_id, 1, detail)
        return False, detail

    _store_source_preview_status(job_manager, job_id, pages, "")
    return True, ""


class ResultResponse(BaseModel):
    """Response model for result endpoint"""
    job_id: str
    status: str
    text: str
    total_pages: int
    source_total_pages: int = 0
    source_preview_error: str = ""
    processed_pages: int
    percentage: float
    engine: Optional[str] = None
    pdf_filename: Optional[str] = None
    sheet_name: Optional[str] = None
    doc_type: Optional[str] = None


class UpdateResultRequest(BaseModel):
    """Request body for updating/editing OCR result text"""
    text: str
    structured: Optional[dict] = None
    layout_pages: Optional[List[dict]] = None

    def __init__(self, **data):
        if "layout_pages" not in data and "layoutPages" in data:
            data["layout_pages"] = data.get("layoutPages")
        super().__init__(**data)


def _page_number(value, default: int = 1) -> int:
    try:
        page = int(value)
        return page if page > 0 else default
    except (TypeError, ValueError):
        return default


def _page_blocks(page_data: dict) -> list[dict]:
    if not isinstance(page_data, dict):
        return []
    blocks = page_data.get("blocks")
    if isinstance(blocks, list):
        return [b for b in blocks if isinstance(b, dict)]
    elements = page_data.get("elements")
    if isinstance(elements, list):
        return [e for e in elements if isinstance(e, dict)]
    return []


def _block_text_for_plain_output(block_type: str, block_text: str, block_html: Optional[str]) -> str:
    if block_type == "table" and block_html:
        try:
            from core.layout_engine import _html_table_to_text
            return _html_table_to_text(block_html)
        except Exception:
            return block_text
    return block_text


def _sync_text_derivatives(job_manager: JobManager, job_id: str, text: str) -> None:
    """Keep plain text and markdown-style text artifacts in lockstep."""
    result_path = job_manager.storage.get_result_file_path(job_id)
    result_path.write_text(text, encoding="utf-8")

    job_dir = job_manager.storage.get_job_directory(job_id)
    try:
        (job_dir / "result.md").write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to sync result.md for job %s: %s", job_id, exc)


async def _reindex_after_save(job_manager: JobManager, job_id: str, text: str, old_text: str) -> None:
    try:
        stats = await job_manager.smart_reindex(job_id, text, old_text=old_text)
        logger.info(
            "Background reindex after save completed for job %s: %s",
            job_id,
            stats,
        )
    except Exception as exc:
        logger.error("Background reindex after save failed for job %s: %s", job_id, exc)


@router.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str) -> ResultResponse:
    """
    Get the OCR result for a completed job

    Args:
        job_id: Job identifier

    Returns:
        ResultResponse with extracted text

    Raises:
        HTTPException: If job not found or not completed
    """
    job_manager = get_job_manager()

    # Check job status
    status_info = job_manager.get_status(job_id)

    if status_info.get('status') == 'not_found':
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Get result text
    result_text = job_manager.get_result(job_id)

    if result_text is None:
        result_text = ""

    return ResultResponse(
        job_id=job_id,
        status=status_info.get('status', 'unknown'),
        text=result_text,
        total_pages=status_info.get('total_pages', 0),
        source_total_pages=status_info.get('source_total_pages', status_info.get('total_pages', 0)),
        source_preview_error=status_info.get('source_preview_error', ""),
        processed_pages=status_info.get('processed_pages', 0),
        percentage=status_info.get('percentage', 0.0),
        engine=status_info.get('engine'),
        pdf_filename=status_info.get('pdf_filename'),
        sheet_name=status_info.get('sheet_name') or None,
        doc_type=status_info.get('doc_type'),
    )


@router.patch("/result/{job_id}")
async def update_result(job_id: str, body: UpdateResultRequest):
    """
    Save edited OCR result text back to disk.

    Overwrites outputs/{job_id}/result.txt with the provided text.
    If structured data is provided, overwrites document.json.
    Keep MongoDB embeddings, per-page JSON files, and result.html perfectly in sync.
    """
    job_manager = get_job_manager()

    status_info = job_manager.get_status(job_id)
    if status_info.get('status') == 'not_found':
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    try:
        # Get old text before we overwrite it to accurately compute changed pages
        old_text = job_manager.storage.read_result(job_id) or ""
        
        # Save structured layout_pages first if provided by rich-text editor
        if body.layout_pages:
            import json
            pages_dir = job_manager.storage.get_job_directory(job_id) / "pages"
            pages_dir.mkdir(exist_ok=True)
            
            incoming_pages = body.layout_pages
            reconstructed_text_parts = []
            
            for p_data in incoming_pages:
                if not isinstance(p_data, dict):
                    continue
                page_num = _page_number(p_data.get("page_number") or p_data.get("page"), 1)
                page_file = pages_dir / f"page_{page_num:04d}.json"
                
                original_data = {}
                if page_file.exists():
                    try:
                        with open(page_file, "r", encoding="utf-8") as f:
                            original_data = json.load(f)
                    except Exception:
                        pass
                
                page_blocks_text = []
                has_blocks_schema = "blocks" in original_data
                
                if has_blocks_schema:
                    updated_blocks = []
                    original_blocks = original_data.get("blocks", [])
                    
                    for idx, b in enumerate(_page_blocks(p_data)):
                        block_id = b.get("block_id") or f"p{page_num}_b{idx:03d}"
                        block_type = b.get("type", "paragraph")
                        block_text = b.get("text", "")
                        block_html = b.get("html")
                        confidence = b.get("confidence", 0.95)
                        reading_order = b.get("reading_order") or (idx + 1)
                        language = b.get("language") or "unknown"
                        
                        new_block = {
                            "block_id": block_id,
                            "type": block_type,
                            "text": block_text,
                            "html": block_html,
                            "reading_order": reading_order,
                            "language": language,
                            "confidence": confidence,
                        }
                        
                        orig_block = next((ob for ob in original_blocks if ob.get("block_id") == block_id), None)
                        if orig_block:
                            new_block.update({
                                "x": orig_block.get("x", 0.0),
                                "y": orig_block.get("y", 0.0),
                                "width": orig_block.get("width", 0.0),
                                "height": orig_block.get("height", 0.0),
                                "font_size": orig_block.get("font_size"),
                                "is_bold": orig_block.get("is_bold"),
                            })
                        else:
                            new_block.update({
                                "x": 0.0,
                                "y": 0.0,
                                "width": 0.0,
                                "height": 0.0,
                            })
                        
                        updated_blocks.append(new_block)
                        page_blocks_text.append(_block_text_for_plain_output(block_type, block_text, block_html))
                    
                    new_page_data = {
                        **original_data,
                        "page_number": page_num,
                        "blocks": updated_blocks,
                    }
                else:
                    updated_elements = []
                    original_elements = original_data.get("elements", [])
                    
                    for idx, b in enumerate(_page_blocks(p_data)):
                        block_id = b.get("block_id") or b.get("id") or f"p{page_num}_b{idx:03d}"
                        block_type = b.get("type", "paragraph")
                        block_text = b.get("text", "")
                        block_html = b.get("html")
                        confidence = b.get("confidence", 0.95)
                        reading_order = b.get("reading_order") or (idx + 1)
                        language = b.get("language") or "unknown"
                        
                        new_element = {
                            "id": block_id,
                            "type": block_type,
                            "text": block_text,
                            "html": block_html,
                            "reading_order": reading_order,
                            "language": language,
                            "confidence": confidence,
                            "cleaned": b.get("cleaned", True)
                        }
                        
                        orig_el = next((oe for oe in original_elements if oe.get("id") == block_id), None)
                        if orig_el:
                            new_element["bbox"] = orig_el.get("bbox")
                        elif b.get("bbox"):
                            new_element["bbox"] = b.get("bbox")
                        else:
                            new_element["bbox"] = [0.0, 0.0, 0.0, 0.0]
                            
                        updated_elements.append(new_element)
                        page_blocks_text.append(_block_text_for_plain_output(block_type, block_text, block_html))
                
                    new_page_data = {
                        **original_data,
                        "page": page_num,
                        "elements": updated_elements,
                    }
                    
                with open(page_file, "w", encoding="utf-8") as f:
                    json.dump(new_page_data, f, ensure_ascii=False, indent=2)
                
                page_text = "\n\n".join(t for t in page_blocks_text if t)
                reconstructed_text_parts.append((page_num, page_text))
                
            reconstructed_text_parts.sort(key=lambda x: x[0])
            full_text_lines = []
            for p_num, p_text in reconstructed_text_parts:
                if full_text_lines:
                    full_text_lines.append("")
                full_text_lines.append(f"--- Page {p_num} ---")
                full_text_lines.append(p_text)
                
            full_rebuilt_text = "\n".join(full_text_lines)
            body.text = full_rebuilt_text
            _sync_text_derivatives(job_manager, job_id, body.text)
            
            if hasattr(job_manager.storage, 'build_combined_structured_result'):
                job_manager.storage.build_combined_structured_result(job_id)
            if hasattr(job_manager.storage, 'build_html_result'):
                job_manager.storage.build_html_result(job_id)
                
            logger.info(f"Successfully processed layout_pages edits and rebuilt derivative files for job {job_id}")
        else:
            # Save the new plain text result to disk first
            _sync_text_derivatives(job_manager, job_id, body.text)
            
            # Save structured data if provided
            if body.structured:
                import json
                pages_dir = job_manager.storage.get_job_directory(job_id) / "pages"
                pages_dir.mkdir(exist_ok=True)
                
                for p_data in body.structured.get("pages", []):
                    if not isinstance(p_data, dict):
                        continue
                    page_num = _page_number(p_data.get("page") or p_data.get("page_number"), 1)
                    page_file = pages_dir / f"page_{page_num:04d}.json"
                    
                    # We need to maintain the original file's structure if it exists
                    original_data = {}
                    if page_file.exists():
                        try:
                            with open(page_file, "r", encoding="utf-8") as f:
                                original_data = json.load(f)
                        except Exception:
                            pass
                    
                    # Convert blocks back to elements
                    elements = []
                    for b in _page_blocks(p_data):
                        elements.append({
                            "type": "paragraph",
                            "id": b.get("id", ""),
                            "text": b.get("text", ""),
                            "bbox": b.get("bbox", []),
                            "confidence": b.get("confidence", 0),
                            "reading_order": b.get("reading_order", 0),
                            "cleaned": b.get("cleaned", False)
                        })
                    
                    new_page_data = {
                        **original_data,
                        "page": page_num,
                        "elements": elements
                    }
                    
                    # Keep metadata like image, width, height if it was sent back
                    if "image" in p_data: new_page_data["image"] = p_data["image"]
                    if "width" in p_data: new_page_data["width"] = p_data["width"]
                    if "height" in p_data: new_page_data["height"] = p_data["height"]
                    
                    with open(page_file, "w", encoding="utf-8") as f:
                        json.dump(new_page_data, f, ensure_ascii=False, indent=2)
                
                # Also save the raw structured payload to document.json for parity with pipeline_v2
                doc_path = job_manager.storage.get_job_directory(job_id) / "document.json"
                with open(doc_path, "w", encoding="utf-8") as f:
                    json.dump(body.structured, f, ensure_ascii=False, indent=2)
                
                # Rebuild the derived files
                if hasattr(job_manager.storage, 'build_combined_structured_result'):
                    job_manager.storage.build_combined_structured_result(job_id)
                if hasattr(job_manager.storage, 'build_html_result'):
                    job_manager.storage.build_html_result(job_id)
                
                logger.info(f"Updated document.json, result.json, and result.html for job {job_id}")
            else:
                # Automap flat text edits back to per-page JSONs and result.html if page JSONs exist
                pages_dir = job_manager.storage.get_job_directory(job_id) / "pages"
                if pages_dir.exists() and any(pages_dir.glob("page_*.json")):
                    import json
                    from core.semantic_search import split_flat_text_pages, split_into_blocks
                    
                    def text_to_html_table(t: str) -> str:
                        tbl_lines = [line.strip() for line in t.split("\n") if line.strip()]
                        if not tbl_lines:
                            return ""
                        html_lines = ["<table>"]
                        for idx, line in enumerate(tbl_lines):
                            cells = line.split("\t")
                            tag = "th" if idx == 0 else "td"
                            row_html = "<tr>" + "".join(f"<{tag}>{cell.strip()}</{tag}>" for cell in cells) + "</tr>"
                            html_lines.append(row_html)
                        html_lines.append("</table>")
                        return "".join(html_lines)

                    updated_pages = split_flat_text_pages(body.text, preserve_tabs=True)
                    if len(updated_pages) == 1 and updated_pages[0][0] is None:
                        updated_pages = [(1, updated_pages[0][1])]
                    
                    for p_num, p_text in updated_pages:
                        if p_num is None:
                            continue
                        page_file = pages_dir / f"page_{p_num:04d}.json"
                        if page_file.exists():
                            try:
                                with open(page_file, "r", encoding="utf-8") as f:
                                    page_data = json.load(f)
                            except Exception:
                                continue
                            
                            original_elements = page_data.get("elements", [])
                            new_blocks = split_into_blocks(p_text, preserve_tabs=True)
                            
                            # Apply sophisticated line-proportional and line-count comparison algorithm
                            new_lines = p_text.split("\n")
                            
                            # Identify active indices and original line counts of elements contributing to flat text
                            active_indices = []
                            original_line_counts = []
                            for idx, el in enumerate(original_elements):
                                if el.get("type") == "table" and el.get("html"):
                                    from core.layout_engine import _html_table_to_text
                                    el_txt = _html_table_to_text(el["html"])
                                    active_indices.append(idx)
                                    original_line_counts.append(len(el_txt.split("\n")) if el_txt else 0)
                                elif el.get("text"):
                                    active_indices.append(idx)
                                    original_line_counts.append(len(el.get("text", "").split("\n")))

                            logger.info(f"JOB {job_id} PAGE {p_num} SYNC DEBUG:")
                            logger.info(f"  original_elements count: {len(original_elements)}")
                            logger.info(f"  new_blocks count: {len(new_blocks)}")
                            logger.info(f"  new_lines count: {len(new_lines)}")
                            logger.info(f"  active_indices: {active_indices}")
                            logger.info(f"  original_line_counts: {original_line_counts} (sum={sum(original_line_counts)})")

                            updated_elements = list(original_elements)
                            
                            # Perfect match: line counts are exactly preserved
                            if len(new_lines) == sum(original_line_counts):
                                logger.info("  -> Line counts MATCH perfectly! Running partition mapping...")
                                current_line_idx = 0
                                for act_idx, line_cnt in zip(active_indices, original_line_counts):
                                    segment_lines = new_lines[current_line_idx : current_line_idx + line_cnt]
                                    current_line_idx += line_cnt
                                    new_val = "\n".join(segment_lines)
                                    
                                    el = updated_elements[act_idx]
                                    el["text"] = new_val
                                    if el.get("type") == "table" and "\t" in new_val:
                                        el["html"] = text_to_html_table(new_val)
                            # Partial match: proportional allocation of lines to elements
                            elif len(original_line_counts) > 0 and sum(original_line_counts) > 0:
                                logger.info("  -> Line counts DO NOT match! Running proportional allocation...")
                                orig_total = sum(original_line_counts)
                                new_total = len(new_lines)
                                
                                allocated_counts = []
                                for cnt in original_line_counts:
                                    if cnt == 0:
                                        allocated_counts.append(0)
                                    else:
                                        share = int(round((cnt / orig_total) * new_total))
                                        share = max(1, share)
                                        allocated_counts.append(share)
                                        
                                # Adjust allocations to match new_total exactly
                                diff = sum(allocated_counts) - new_total
                                if diff > 0:
                                    while diff > 0:
                                        candidates = [i for i, c in enumerate(allocated_counts) if c > 1]
                                        if not candidates:
                                            break
                                        max_idx = max(candidates, key=lambda i: allocated_counts[i])
                                        allocated_counts[max_idx] -= 1
                                        diff -= 1
                                elif diff < 0:
                                    while diff < 0:
                                        max_idx = max(range(len(allocated_counts)), key=lambda i: allocated_counts[i])
                                        allocated_counts[max_idx] += 1
                                        diff += 1
                                        
                                current_line_idx = 0
                                for act_idx, line_cnt in zip(active_indices, allocated_counts):
                                    segment_lines = new_lines[current_line_idx : current_line_idx + line_cnt]
                                    current_line_idx += line_cnt
                                    new_val = "\n".join(segment_lines)
                                    
                                    el = updated_elements[act_idx]
                                    el["text"] = new_val
                                    if el.get("type") == "table" and "\t" in new_val:
                                        el["html"] = text_to_html_table(new_val)
                            # Absolute fallback: sequential or structural block matching
                            else:
                                logger.info("  -> No active original lines! Running sequential block matching fallback...")
                                updated_elements = []
                                for i, new_txt in enumerate(new_blocks):
                                    if i < len(original_elements):
                                        el = original_elements[i]
                                        el["text"] = new_txt
                                        if el.get("type") == "table" and "\t" in new_txt:
                                            el["html"] = text_to_html_table(new_txt)
                                        updated_elements.append(el)
                                    else:
                                        updated_elements.append({
                                            "type": "paragraph",
                                            "id": f"p{p_num}_added_{i}",
                                            "text": new_txt,
                                            "bbox": [100, 100, 900, 900],
                                            "confidence": 1.0,
                                            "reading_order": i + 1,
                                            "cleaned": True
                                        })
                            
                            page_data["elements"] = updated_elements
                            with open(page_file, "w", encoding="utf-8") as f:
                                json.dump(page_data, f, ensure_ascii=False, indent=2)
                    
                    if hasattr(job_manager.storage, 'build_combined_structured_result'):
                        job_manager.storage.build_combined_structured_result(job_id)
                    if hasattr(job_manager.storage, 'build_html_result'):
                        job_manager.storage.build_html_result(job_id)
                    
                    logger.info(f"Automatically synchronized page JSONs and rebuilt result.html/result.json for job {job_id}")
            
        # Re-index in the background so the Save button is released as soon as
        # all file artifacts have been written and rebuilt.
        asyncio.create_task(_reindex_after_save(job_manager, job_id, body.text, old_text))
        logger.info(f"Updated result artifacts for job {job_id} ({len(body.text)} chars); reindex scheduled")
    except Exception as e:
        logger.error(f"Failed to save edited result for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save result: {str(e)}"
        )

    return {
        "job_id": job_id, 
        "saved": True, 
        "length": len(body.text),
        "reindex_status": "scheduled",
    }


@router.get("/result/{job_id}/text", response_class=PlainTextResponse)
async def get_result_text(job_id: str) -> str:
    """
    Get the OCR result as plain text
    
    Args:
        job_id: Job identifier
        
    Returns:
        Plain text result
        
    Raises:
        HTTPException: If job not found
    """
    job_manager = get_job_manager()
    
    result_text = job_manager.get_result(job_id)
    
    if result_text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or no result available"
        )
    
    return result_text


@router.get("/result/{job_id}/structured")
async def get_structured_result(job_id: str):
    """
    Get the structured OCR result (per-page elements with types, bboxes, etc.).
    Returns the full result.json built after layout-mode processing.

    Falls back to a minimal wrapper around plain text if no structured data.
    """
    job_manager = get_job_manager()

    status_info = job_manager.get_status(job_id)
    if status_info.get('status') == 'not_found':
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Try structured first
    structured = None
    if hasattr(job_manager.storage, 'read_structured_result'):
        structured = job_manager.storage.read_structured_result(job_id)

    if structured:
        return JSONResponse(content=structured)

    # Fallback: wrap plain text
    result_text = job_manager.get_result(job_id)
    if result_text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} result not available"
        )

    return JSONResponse(content={
        "total_pages": status_info.get('total_pages', 0),
        "pages": [{
            "page": 1,
            "elements": [{"type": "paragraph", "text": result_text}]
        }],
        "_note": "Flat text fallback — structured data not available for this job"
    })


@router.get("/result/{job_id}/layout")
async def get_layout_result(job_id: str):
    """
    Get unified layout blocks page-by-page from page JSONs.
    """
    job_manager = get_job_manager()

    status_info = job_manager.get_status(job_id)
    if status_info.get('status') == 'not_found':
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    pages_dir = job_manager.storage.get_job_directory(job_id) / "pages"
    if not pages_dir.exists():
        # Rebuild layout on the fly from flat text if missing
        result_text = job_manager.get_result(job_id) or ""
        return JSONResponse(content={
            "pages": [{
                "page_number": 1,
                "blocks": [{
                    "block_id": "p1_b000",
                    "type": "paragraph",
                    "text": result_text,
                    "confidence": 1.0,
                    "reading_order": 1,
                    "language": "unknown"
                }]
            }]
        })

    page_files = sorted(pages_dir.glob("page_*.json"))
    if not page_files:
        result_text = job_manager.get_result(job_id) or ""
        return JSONResponse(content={
            "pages": [{
                "page_number": 1,
                "blocks": [{
                    "block_id": "p1_b000",
                    "type": "paragraph",
                    "text": result_text,
                    "confidence": 1.0,
                    "reading_order": 1,
                    "language": "unknown"
                }]
            }]
        })

    import json
    unified_pages = []
    
    for pf in page_files:
        try:
            with open(pf, "r", encoding="utf-8") as f:
                page_data = json.load(f)
        except Exception:
            continue
            
        page_num = page_data.get("page_number") or page_data.get("page", 1)
        width_px = page_data.get("width_px") or page_data.get("width", 0)
        height_px = page_data.get("height_px") or page_data.get("height", 0)
        image = page_data.get("image")
        
        unified_blocks = []
        
        # Schema A: blocks list
        if "blocks" in page_data:
            for idx, b in enumerate(page_data.get("blocks", [])):
                block_id = b.get("block_id") or f"p{page_num}_b{idx:03d}"
                block_type = b.get("type", "paragraph")
                text = b.get("text", "")
                html = b.get("html")
                confidence = b.get("confidence", 0.0)
                reading_order = b.get("reading_order") or (idx + 1)
                language = b.get("language") or "unknown"
                
                # Get coordinates
                x = b.get("x")
                y = b.get("y")
                w = b.get("width")
                h = b.get("height")
                bbox = b.get("bbox")
                if bbox is None and x is not None and y is not None and w is not None and h is not None:
                    if x > 0 or y > 0 or w > 0 or h > 0:
                        bbox = [x, y, x + w, y + h]
                        
                unified_blocks.append({
                    "block_id": block_id,
                    "type": block_type,
                    "text": text,
                    "html": html,
                    "reading_order": reading_order,
                    "language": language,
                    "confidence": confidence,
                    "bbox": bbox
                })
        # Schema B: elements list
        elif "elements" in page_data:
            for idx, el in enumerate(page_data.get("elements", [])):
                block_id = el.get("id") or el.get("block_id") or f"p{page_num}_b{idx:03d}"
                block_type = el.get("type", "paragraph")
                text = el.get("text", "")
                html = el.get("html")
                confidence = el.get("confidence", 0.0)
                reading_order = el.get("reading_order") or (idx + 1)
                language = el.get("language") or "unknown"
                bbox = el.get("bbox")
                
                unified_blocks.append({
                    "block_id": block_id,
                    "type": block_type,
                    "text": text,
                    "html": html,
                    "reading_order": reading_order,
                    "language": language,
                    "confidence": confidence,
                    "bbox": bbox
                })
                
        unified_pages.append({
            "page_number": page_num,
            "width_px": width_px,
            "height_px": height_px,
            "image": image,
            "blocks": unified_blocks
        })
        
    return JSONResponse(content={"pages": unified_pages})


@router.get("/result/{job_id}/html", response_class=HTMLResponse)
async def get_html_result(job_id: str):
    """
    Get the OCR result rendered as HTML (preserves tables, headings, etc.).
    """
    job_manager = get_job_manager()

    status_info = job_manager.get_status(job_id)
    if status_info.get('status') == 'not_found':
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # If it's an Excel job, dynamically generate the HTML table from the source file
    if status_info.get('engine') == 'excel':
        pdf_filename = status_info.get('pdf_filename')
        if pdf_filename:
            source_path = Path(settings.upload_dir) / pdf_filename
            if source_path.exists():
                try:
                    import pandas as pd
                    # Load the specific sheet that was processed
                    sheet_name = status_info.get('sheet_name')
                    logger.info(f"Dynamically building HTML for excel job {job_id}, sheet: {sheet_name!r}")
                    if sheet_name:
                        try:
                            df = pd.read_excel(str(source_path), sheet_name=sheet_name, dtype=str)
                            logger.info(f"Successfully loaded sheet {sheet_name!r}, shape: {df.shape}")
                        except Exception as e:
                            logger.error(f"Failed to load sheet {sheet_name!r}: {e}. Falling back to first sheet.")
                            df = pd.read_excel(str(source_path), dtype=str)
                    else:
                        logger.info("No sheet_name found in status, loading first sheet.")
                        df = pd.read_excel(str(source_path), dtype=str)
                        
                    df.fillna("", inplace=True)
                    table_html = df.to_html(index=False, classes="spreadsheet-table")
                    
                    styled_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>Excel Preview</title>
                        <style>
                            html, body {{ height: 100%; }}
                            body {{ font-family: Calibri, "Segoe UI", Arial, sans-serif; background-color: #f8fafc; color: #111827; margin: 0; overflow: auto; }}
                            .sheet-wrap {{ min-width: max-content; padding: 12px; }}
                            .spreadsheet-table {{ border-collapse: collapse; background: white; font-size: 12px; table-layout: auto; }}
                            .spreadsheet-table th, .spreadsheet-table td {{ border: 1px solid #cbd5e1; padding: 5px 8px; text-align: left; white-space: nowrap; vertical-align: top; }}
                            .spreadsheet-table th {{ background-color: #e5edf7; font-weight: 700; color: #111827; position: sticky; top: 0; z-index: 1; }}
                            .spreadsheet-table tr:nth-child(even) {{ background-color: #f8fafc; }}
                            .spreadsheet-table tr:hover {{ background-color: #eef6ff; }}
                        </style>
                    </head>
                    <body>
                        <div class="sheet-wrap">{table_html}</div>
                    </body>
                    </html>
                    """
                    return HTMLResponse(content=styled_html)
                except Exception as e:
                    logger.error(f"Failed to generate Excel HTML preview: {e}")

    # Try pre-built HTML first
    html_path = None
    if hasattr(job_manager.storage, 'get_html_result_path'):
        html_path = job_manager.storage.get_html_result_path(job_id)

    if html_path and html_path.exists():
        html_content = html_path.read_text(encoding='utf-8')
        return HTMLResponse(content=html_content)

    # Generate on the fly if structured data exists
    if hasattr(job_manager.storage, 'build_html_result'):
        html_str = job_manager.storage.build_html_result(job_id)
        if html_str:
            return HTMLResponse(content=html_str)

    # Fallback: plain text in a <pre> block
    result_text = job_manager.get_result(job_id)
    if result_text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} result not available"
        )

    return HTMLResponse(content=f"<html><body><pre>{result_text}</pre></body></html>")


@router.get("/source/{job_id}")
async def get_source_pdf(job_id: str):
    """
    Serve a PDF source preview for inline viewing.

    Returns:
        The original PDF, or a generated PDF preview for Office sources.

    Raises:
        HTTPException: If job not found or PDF file not available.
    """
    job_manager = get_job_manager()

    status_info = job_manager.get_status(job_id)
    if status_info.get('status') == 'not_found':
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    pdf_filename = status_info.get('pdf_filename')
    if not pdf_filename:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No source PDF associated with job {job_id}"
        )

    source_path = None
    if hasattr(job_manager.storage, 'get_source_pdf_path'):
        source_path = job_manager.storage.get_source_pdf_path(job_id)

    # Try to retrieve from cloud storage if missing locally
    if source_path and not source_path.exists():
        logger.info(f"Source preview for {pdf_filename} missing locally, attempting to download from storage")
        if hasattr(job_manager.storage, 'download_source_pdf'):
            job_manager.storage.download_source_pdf(job_id, source_path)

    if source_path and status_info.get("engine") in {"word", "excel"}:
        if not source_path.exists() or not _looks_like_pdf(source_path):
            logger.info("Regenerating Office source preview for job %s", job_id)
            repaired, repair_error = _repair_office_source_preview(
                job_manager,
                job_id,
                pdf_filename,
                source_path,
            )
            if not repaired:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=repair_error,
                )
            status_info = job_manager.get_status(job_id)

    # Backward compatibility for old PDF jobs created before job-scoped source.pdf.
    if (not source_path or not source_path.exists()) and str(pdf_filename).lower().endswith(".pdf"):
        upload_pdf_path = Path(settings.upload_dir) / pdf_filename
        if upload_pdf_path.exists():
            source_path = upload_pdf_path

    if not source_path or not source_path.exists():
        detail = status_info.get("source_preview_error") or f"Source PDF preview for {pdf_filename} not found"
        code = status.HTTP_409_CONFLICT if status_info.get("engine") in {"word", "excel"} else status.HTTP_404_NOT_FOUND
        raise HTTPException(
            status_code=code,
            detail=detail,
        )

    if not _looks_like_pdf(source_path):
        detail = status_info.get("source_preview_error") or "Source preview is not a valid PDF"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        )

    return FileResponse(
        path=str(source_path),
        media_type='application/pdf',
        filename=pdf_filename,
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, max-age=3600",
        }
    )


@router.get("/result/{job_id}/download")
async def download_result(job_id: str):
    """
    Download the OCR result as a text file
    
    Args:
        job_id: Job identifier
        
    Returns:
        PlainTextResponse with attachment header
        
    Raises:
        HTTPException: If job not found
    """
    job_manager = get_job_manager()
    
    result_text = job_manager.get_result(job_id)
    
    if result_text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} result not found or not available"
        )
    
    return PlainTextResponse(
        content=result_text,
        media_type='text/plain',
        headers={"Content-Disposition": f"attachment; filename=ocr_result_{job_id}.txt"}
    )


@router.get("/pages/{job_id}/{filename}")
async def get_page_image(job_id: str, filename: str):
    """
    Serve a page image from the job's pages directory.

    Only allows image files (png, jpg, jpeg) to prevent path traversal.
    """
    import re
    from pathlib import Path
    from fastapi.responses import FileResponse

    # Security: only allow safe filenames
    if not re.match(r'^page_\d+\.(png|jpg|jpeg)$', filename, re.IGNORECASE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )

    job_manager = get_job_manager()
    pages_dir = job_manager.storage.get_job_directory(job_id) / "pages"
    image_path = pages_dir / filename

    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page image {filename} not found"
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )
