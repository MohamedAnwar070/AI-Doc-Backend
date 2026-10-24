"""Source document preview generation.

PDF uploads can be shown directly. Office formats need a PDF preview for the
browser's built-in viewer and for reliable page navigation. LibreOffice's
headless converter is the standard server-side route for that.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class SourcePreviewError(RuntimeError):
    """Raised when a source preview cannot be generated."""


def find_soffice() -> str | None:
    """Locate LibreOffice's ``soffice`` executable."""
    if settings.soffice_path:
        configured = Path(settings.soffice_path)
        if configured.exists():
            return str(configured)
        found = shutil.which(settings.soffice_path)
        if found:
            return found

    for name in ("soffice", "soffice.exe", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found

    for candidate in (
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
    ):
        if candidate.exists():
            return str(candidate)

    return None


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF, falling back to 1 on read errors."""
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            return max(len(doc), 1)
    except Exception:
        logger.debug("Could not count pages in source preview %s", pdf_path, exc_info=True)
        return 1


def extract_pdf_text_pages(pdf_path: Path) -> list[dict]:
    """Extract simple per-page text elements from a text-layer PDF preview."""
    pages: list[dict] = []
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                elements: list[dict] = []
                blocks = page.get_text("blocks")
                for block_index, block in enumerate(blocks):
                    x0, y0, x1, y1, text, *_ = block
                    text = (text or "").strip()
                    if not text:
                        continue
                    elements.append({
                        "id": f"pdf-preview-p{page_index + 1}-b{block_index}",
                        "type": "paragraph",
                        "text": text,
                        "html": f"<p>{html_escape(text)}</p>",
                        "bbox": [x0, y0, x1, y1],
                        "reading_order": block_index,
                        "confidence": 1.0,
                        "cleaned": True,
                    })

                pages.append({
                    "page": page_index + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "elements": elements,
                })
    except Exception:
        logger.debug("Could not extract text pages from preview %s", pdf_path, exc_info=True)
        return []

    return pages


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def _create_microsoft_office_pdf_preview(source_path: Path, target_pdf_path: Path) -> tuple[Path, int]:
    """Convert Office documents to PDF via Microsoft Office on Windows."""
    if platform.system().lower() != "windows":
        raise SourcePreviewError("Microsoft Office PDF conversion is only available on Windows")

    suffix = source_path.suffix.lower()
    try:
        import pythoncom
        import win32com.client
    except Exception as exc:
        logger.info("pywin32 unavailable; falling back to PowerShell Office automation")
        return _create_microsoft_office_pdf_preview_with_powershell(source_path, target_pdf_path)

    pythoncom.CoInitialize()
    app = None
    doc = None
    try:
        target_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".docx":
            app = win32com.client.DispatchEx("Word.Application")
            app.Visible = False
            doc = app.Documents.Open(str(source_path), ReadOnly=True)
            doc.ExportAsFixedFormat(str(target_pdf_path), 17)
        elif suffix == ".xlsx":
            app = win32com.client.DispatchEx("Excel.Application")
            app.Visible = False
            app.DisplayAlerts = False
            doc = app.Workbooks.Open(str(source_path), ReadOnly=True)
            doc.ExportAsFixedFormat(0, str(target_pdf_path))
        else:
            raise SourcePreviewError(f"Unsupported Office preview type: {suffix}")
    except Exception as exc:
        raise SourcePreviewError(f"Microsoft Office conversion failed: {exc}") from exc
    finally:
        try:
            if doc is not None:
                if suffix == ".docx":
                    doc.Close(False)
                else:
                    doc.Close(SaveChanges=False)
        except Exception:
            logger.debug("Failed to close Office document after preview conversion", exc_info=True)
        try:
            if app is not None:
                app.Quit()
        except Exception:
            logger.debug("Failed to quit Office application after preview conversion", exc_info=True)
        pythoncom.CoUninitialize()

    if not target_pdf_path.exists():
        raise SourcePreviewError("Microsoft Office did not produce a PDF preview")

    return target_pdf_path, count_pdf_pages(target_pdf_path)


def _ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _create_microsoft_office_pdf_preview_with_powershell(
    source_path: Path,
    target_pdf_path: Path,
) -> tuple[Path, int]:
    """Convert Office documents to PDF using PowerShell COM automation."""
    suffix = source_path.suffix.lower()
    target_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    source = _ps_single_quote(str(source_path))
    target = _ps_single_quote(str(target_pdf_path))

    if suffix == ".docx":
        script = f"""
$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
try {{
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $doc = $word.Documents.Open({source}, $false, $true)
  $doc.ExportAsFixedFormat({target}, 17)
}} finally {{
  if ($doc -ne $null) {{ $doc.Close($false) | Out-Null }}
  if ($word -ne $null) {{ $word.Quit() | Out-Null }}
}}
"""
    elif suffix == ".xlsx":
        script = f"""
$ErrorActionPreference = 'Stop'
$excel = $null
$book = $null
try {{
  $excel = New-Object -ComObject Excel.Application
  $excel.Visible = $false
  $excel.DisplayAlerts = $false
  $book = $excel.Workbooks.Open({source}, 3, $true)
  $book.ExportAsFixedFormat(0, {target})
}} finally {{
  if ($book -ne $null) {{ $book.Close($false) | Out-Null }}
  if ($excel -ne $null) {{ $excel.Quit() | Out-Null }}
}}
"""
    else:
        raise SourcePreviewError(f"Unsupported Office preview type: {suffix}")

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=settings.office_preview_timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        raise SourcePreviewError(
            f"Microsoft Office conversion failed: {completed.stderr or completed.stdout}"
        )
    if not target_pdf_path.exists():
        raise SourcePreviewError("Microsoft Office did not produce a PDF preview")

    return target_pdf_path, count_pdf_pages(target_pdf_path)


def create_office_pdf_preview(source_path: Path, target_pdf_path: Path) -> tuple[Path, int]:
    """Convert a DOCX/XLSX source file to PDF for browser preview."""
    source_path = source_path.resolve()
    target_pdf_path = target_pdf_path.resolve()

    soffice = find_soffice()
    if not soffice:
        return _create_microsoft_office_pdf_preview(source_path, target_pdf_path)

    target_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="office-preview-") as tmp:
        out_dir = Path(tmp)
        command = [
            soffice,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(source_path),
        ]
        logger.info("Generating source preview with LibreOffice: %s", source_path.name)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=settings.office_preview_timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise SourcePreviewError(
                f"LibreOffice conversion failed: {completed.stderr or completed.stdout}"
            )

        converted = out_dir / f"{source_path.stem}.pdf"
        if not converted.exists():
            pdfs = list(out_dir.glob("*.pdf"))
            converted = pdfs[0] if pdfs else converted
        if not converted.exists():
            raise SourcePreviewError("LibreOffice did not produce a PDF preview")

        shutil.copy2(converted, target_pdf_path)

    return target_pdf_path, count_pdf_pages(target_pdf_path)
