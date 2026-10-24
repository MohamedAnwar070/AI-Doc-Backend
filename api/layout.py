"""Layout API — exposes spec §3 ``LayoutDocument`` and a preview HTML.

Endpoints
---------
* ``GET /api/layout/{job_id}`` — JSON :class:`~ocr.layout_schema.LayoutDocument`
* ``GET /api/layout/{job_id}/preview`` — Self-contained HTML rendering of
  the layout using absolute-positioned blocks. Useful for validation while
  the React frontend renderer is built.

Both endpoints read the per-page JSONs already produced by the OCR worker
(``outputs/<job_id>/pages/page_NNNN.json``) and the source PDF
(``outputs/<job_id>/source.pdf``) for true page dimensions.
"""
from __future__ import annotations

import html
import logging
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse

from core.job_manager import get_job_manager
from ocr.layout_builder import build_from_disk
from ocr.layout_schema import LayoutDocument, LayoutPage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/layout", tags=["layout"])


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_layout(job_id: str) -> LayoutDocument:
    job_manager = get_job_manager()
    storage = job_manager.storage

    if not storage.job_exists(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    job_dir: Path = storage.get_job_directory(job_id)
    pages_dir = job_dir / "pages"
    pdf_path: Path = storage.get_source_pdf_path(job_id)

    status_info = job_manager.get_status(job_id)
    source_file = status_info.get("pdf_filename") or pdf_path.name

    if not pages_dir.exists() or not any(pages_dir.glob("page_*.json")):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No structured pages available for job {job_id}. "
                "Wait for OCR to complete or re-run with use_layout enabled."
            ),
        )

    return build_from_disk(
        job_id=job_id,
        source_file=source_file,
        pages_dir=pages_dir,
        pdf_path=pdf_path if pdf_path.exists() else None,
    )


# ── JSON endpoint ──────────────────────────────────────────────────────────


@router.get(
    "/{job_id}",
    response_model=LayoutDocument,
    summary="Get spec §3 LayoutDocument for a job",
)
async def get_layout(job_id: str) -> LayoutDocument:
    return _build_layout(job_id)


# ── HTML preview endpoint ──────────────────────────────────────────────────


_PREVIEW_CSS = """
:root { color-scheme: light; }
body { background: #f3f4f6; margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
header { padding: 12px 24px; background: #111827; color: #fff;
         display: flex; gap: 16px; align-items: center; }
header h1 { font-size: 16px; font-weight: 600; margin: 0; }
header .meta { font-size: 12px; opacity: 0.8; }
.pages { padding: 24px; display: flex; flex-direction: column; gap: 32px;
         align-items: center; }
.page { background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        position: relative; transform-origin: top left; background-size: 100% 100%; }
.page.source-image .block { color: transparent; }
.page.source-image .block:hover { color: #111827; background: rgba(255,255,255,0.92); }
.block { position: absolute; padding: 2px 4px; box-sizing: border-box;
         font-size: 12px; line-height: 1.35; white-space: pre-wrap;
         border: 1px solid transparent; }
.block:hover { border-color: #3b82f6; background: rgba(59,130,246,0.05); }
.block.heading, .block.title { font-weight: 700; color: #111827; }
.block.table { font-family: 'Consolas', monospace; font-size: 11px; }
.block.low-confidence {
  border: 2px solid #f59e0b;
  background: rgba(245,158,11,0.08);
}
.block .badge { position: absolute; top: -10px; left: 0;
                font-size: 9px; padding: 1px 5px; background: #111827;
                color: #fff; border-radius: 3px; opacity: 0; transition: 0.15s; }
.block:hover .badge { opacity: 1; }
"""


_PREVIEW_JS = """
// Auto-fit each page to viewport width by scaling down.
function fit() {
  const maxW = Math.min(window.innerWidth - 64, 1200);
  document.querySelectorAll('.page').forEach(p => {
    const w = parseFloat(p.dataset.width);
    const scale = Math.min(1, maxW / w);
    p.style.transform = `scale(${scale})`;
    p.style.marginBottom = `${(parseFloat(p.dataset.height) * scale) + 32 - parseFloat(p.dataset.height)}px`;
  });
}
window.addEventListener('resize', fit); fit();
"""


def _render_block_html(block) -> str:
    classes = ["block", html.escape(block.type)]
    if block.confidence and block.confidence < 0.85:
        classes.append("low-confidence")
    style = (
        f"left:{block.x:.1f}px;"
        f"top:{block.y:.1f}px;"
        f"width:{block.width:.1f}px;"
        f"height:{block.height:.1f}px;"
    )
    if block.font_size:
        style += f"font-size:{float(block.font_size):.1f}px;"
    if block.is_bold:
        style += "font-weight:700;"

    badge = (
        f"<span class='badge'>{html.escape(block.block_id)} "
        f"({block.confidence:.2f})</span>"
        if block.confidence
        else f"<span class='badge'>{html.escape(block.block_id)}</span>"
    )

    if block.html:
        # Tables: pass through HTML, plus our positioning wrapper
        body = block.html
    else:
        body = html.escape(block.text)

    return (
        f'<div class="{" ".join(classes)}" '
        f'data-block-id="{html.escape(block.block_id)}" '
        f'data-confidence="{block.confidence:.3f}" '
        f'data-reading-order="{block.reading_order}" '
        f'style="{style}">{badge}{body}</div>'
    )


def _render_page_html(page: LayoutPage, job_id: str) -> str:
    blocks_html = "\n".join(_render_block_html(b) for b in page.blocks)
    image_ref = getattr(page, "image", None)
    page_class = "page"
    background = ""
    if image_ref:
        filename = Path(str(image_ref)).name
        page_class += " source-image"
        background = (
            f"background-image:url('/api/pages/{quote(job_id, safe='')}/"
            f"{quote(filename, safe='')}');"
        )
    return (
        f'<div class="{page_class}" data-page="{page.page_number}" '
        f'data-width="{page.width_px}" data-height="{page.height_px}" '
        f'style="width:{page.width_px}px; height:{page.height_px}px;{background}">'
        f"{blocks_html}</div>"
    )


@router.get(
    "/{job_id}/preview",
    response_class=HTMLResponse,
    summary="Self-contained HTML preview using LayoutDocument",
)
async def get_layout_preview(job_id: str) -> HTMLResponse:
    layout = _build_layout(job_id)

    pages_html = "\n".join(_render_page_html(p, job_id) for p in layout.pages)
    source = html.escape(layout.source_file)

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Layout preview — {source}</title>
<style>{_PREVIEW_CSS}</style>
</head>
<body>
<header>
  <h1>Layout preview</h1>
  <div class="meta">job <code>{html.escape(job_id)}</code> · {source} · {layout.total_pages} pages · method={layout.extraction_method}</div>
</header>
<main class="pages">{pages_html}</main>
<script>{_PREVIEW_JS}</script>
</body>
</html>
"""
    return HTMLResponse(content=body)


@router.get(
    "/{job_id}/raw",
    summary="Raw aggregated layout JSON (no Pydantic validation)",
)
async def get_layout_raw(job_id: str) -> JSONResponse:
    layout = _build_layout(job_id)
    return JSONResponse(layout.model_dump(mode="json"))


__all__ = ["router"]
