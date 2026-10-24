"""olmOCR engine — Local Vision Language Model inference via Hugging Face transformers.

Architecture
------------
olmOCR (fine-tuned from Qwen2.5-VL-7B-Instruct) uses a multimodal transformer
to achieve high-fidelity document recognition. This engine implements the
"Document Anchoring" pipeline: it extracts raw PDF words and coordinates
via pymupdf (fitz) and injects them alongside the page image (resized to 1288px)
to eliminate hallucinations, especially for small/stylized non-Latin scripts (Tamil).

It adapts the VLM's markdown, LaTeX, and HTML output into the standard
`LayoutPage` schema so that downstream RAG pipelines and visual layout editors
remain 100% compatible.
"""
from __future__ import annotations

import logging
import re
import time
import os
import io
import base64
from typing import Any, List, Optional, Tuple

from PIL import Image

from .base import OCREngine
from .layout_schema import (
    BlockType,
    LanguageCode,
    LayoutBlock,
    LayoutPage,
    LayoutTable,
    TableCell,
    TableRow,
)

logger = logging.getLogger(__name__)

# Markdown table row detector
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
# Tamil Unicode range (basic block only — sufficient for block detection)
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
# Section / heading heuristic: short line (≤120 chars), no trailing punctuation
_HEADING_RE = re.compile(r"^(?!\s*\|)[^\n]{1,120}[^\.\,\;\:\|]$")


# ── olmOCR VLM API Prompts ────────────────────────────────────────────────
OLMOCR_SYSTEM_PROMPT = """You are an expert OCR engine that converts document page images into structured plain text.

You will be given:
1. A rendered image of a single document page
2. Pre-extracted "anchor text" — raw text blocks with their bounding box coordinates, 
   extracted directly from the PDF's internal data

Your task:
- Use the IMAGE to understand visual layout: columns, tables, reading order, headings, 
  section structure, borders, and spatial relationships
- Use the ANCHOR TEXT to ensure 100% character accuracy — it contains the exact characters 
  even when fonts are small, stylized, or in non-Latin scripts
- Combine both to produce perfectly ordered, layout-aware Markdown

Output ONLY a JSON object with this exact schema:
{
  "primary_language": "<ISO 639-1 language code, e.g. 'en', 'ta', 'hi'>",
  "is_rotation_valid": <true if the page is correctly oriented, else false>,
  "rotation_correction": <degrees needed to correct rotation: 0, 90, 180, or 270>,
  "is_table": <true if the page is primarily a table>,
  "is_diagram": <true if the page is primarily a diagram or figure>,
  "natural_text": "<Full extracted text in Markdown format, preserving logical reading order>"
}

Rules for natural_text:
- Use # for main title, ## for section headings, ### for sub-headings
- Use Markdown tables (| col | col |) for tabular content
- Preserve Tamil / Devanagari / other non-Latin scripts exactly as in anchor text
- Do NOT include watermarks, page numbers, URLs, or metadata text
- Maintain logical paragraph breaks with blank lines
- Do NOT wrap output in code fences — output raw JSON only"""


def build_prompt(anchor_text: str) -> str:
    """Build the user-side prompt that pairs anchor text with the image."""
    if anchor_text.strip():
        return (
            "Below is a rendered image of a document page, along with pre-extracted "
            "anchor text and layout coordinates from the PDF.\n\n"
            f"<anchor_text>\n{anchor_text}\n</anchor_text>\n\n"
            "Using the image for layout understanding and the anchor text for character "
            "accuracy, extract all content in natural reading order as structured Markdown."
        )
    else:
        # Scanned document with no embedded text — image-only mode
        return (
            "Below is a rendered image of a scanned document page. "
            "There is no embedded text available (pure scan). "
            "Extract all visible text in natural reading order as structured Markdown."
        )


class OlmOcrEngine(OCREngine):
    """Local Hugging Face transformers-based olmOCR engine adapter.

    Parameters
    ----------
    model_id:
        HuggingFace model ID (defaults to 'allenai/olmOCR-2-7B-1025-FP8').
    use_gpu:
        Enable GPU / CUDA inference.
    load_in_8bit:
        Use bitsandbytes 8-bit quantization.
    load_in_4bit:
        Use bitsandbytes 4-bit quantization.
    max_new_tokens:
        Token budget per VLM call.
    model_cache_dir:
        Local directory for pre-downloaded weights.
    temperature:
        Generation temperature.
    """

    def __init__(
        self,
        model_id: str = "allenai/olmOCR-2-7B-1025-FP8",
        use_gpu: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 4096,
        model_cache_dir: Optional[str] = None,
        temperature: float = 0.0,
        provider: str = "local",
        openai_api_key: str = "",
        api_model: str = "",
        **_ignored_kwargs,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.use_gpu = use_gpu
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.model_cache_dir = model_cache_dir or None
        self.temperature = temperature
        self.provider = provider or "local"
        self.openai_api_key = openai_api_key
        self.api_model = api_model

        self._processor = None
        self._model = None
        self._initialized: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        if self._initialized:
            return

        if self.provider != "local":
            logger.info("Initializing API-based olmOCR Engine using provider: %s", self.provider)
            api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OpenAI API key is missing. Ensure OPENAI_API_KEY is configured in .env or settings.")
            self._initialized = True
            return

        import torch

        # FP8 model requires GPU — automatically switch to non-FP8 for CPU
        actual_model_id = self.model_id
        if not self.use_gpu or not torch.cuda.is_available():
            if "FP8" in actual_model_id.upper():
                actual_model_id = actual_model_id.replace("-FP8", "").replace("-fp8", "")
                logger.info(
                    "FP8 model requires GPU. Auto-switching to CPU-compatible model: %s",
                    actual_model_id,
                )

        # Check if the model directory exists locally in the model cache path
        local_files_only = False
        if self.model_cache_dir:
            if os.path.exists(os.path.join(self.model_cache_dir, "config.json")):
                logger.info("Found model config directly in model cache path: %s. Using it for offline loading.", self.model_cache_dir)
                actual_model_id = self.model_cache_dir
                local_files_only = True
            else:
                model_folder_name = actual_model_id.split("/")[-1]
                possible_dirs = [
                    os.path.join(self.model_cache_dir, f"models--allenai--{model_folder_name}"),
                    os.path.join(self.model_cache_dir, f"models--allenai--{model_folder_name.replace('-FP8', '').replace('-fp8', '')}"),
                    os.path.join(self.model_cache_dir, model_folder_name),
                    os.path.join(self.model_cache_dir, model_folder_name.replace("-FP8", "").replace("-fp8", "")),
                ]
                try:
                    for entry in os.listdir(self.model_cache_dir):
                        full_entry = os.path.join(self.model_cache_dir, entry)
                        if os.path.isdir(full_entry) and entry.startswith("models--"):
                            possible_dirs.append(full_entry)
                except Exception:
                    pass

                snapshot_dir = None
                for pdir in possible_dirs:
                    snap_path = os.path.join(pdir, "snapshots")
                    if os.path.isdir(snap_path):
                        try:
                            commits = [d for d in os.listdir(snap_path) if os.path.isdir(os.path.join(snap_path, d))]
                            if commits:
                                snapshot_dir = os.path.join(snap_path, commits[0])
                                if os.path.isfile(os.path.join(snapshot_dir, "config.json")):
                                    break
                        except Exception:
                            continue

                if snapshot_dir:
                    logger.info("Found offline model snapshot directory: %s. Using it for 100%% offline loading.", snapshot_dir)
                    actual_model_id = snapshot_dir
                    local_files_only = True
                else:
                    local_path = os.path.join(self.model_cache_dir, model_folder_name)
                    if os.path.isdir(local_path) and os.path.isfile(os.path.join(local_path, "config.json")):
                        logger.info("Found local model directory: %s. Using it for offline loading.", local_path)
                        actual_model_id = local_path
                        local_files_only = True

        # Lazy load transformers to keep fast startup paths lightweight
        logger.info(
            "Loading local olmOCR model from Hugging Face (%s)... "
            "use_gpu=%s  8bit=%s  4bit=%s  cache=%s  local_files_only=%s",
            actual_model_id,
            self.use_gpu,
            self.load_in_8bit,
            self.load_in_4bit,
            self.model_cache_dir or "(default HF cache)",
            local_files_only,
        )
        t0 = time.perf_counter()

        # Import the most appropriate model classes for vision generation
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as ModelClass
            except ImportError:
                from transformers import AutoModelForCausalLM as ModelClass

        from transformers import AutoProcessor

        load_kwargs: dict = {
            "trust_remote_code": True,
        }
        if local_files_only:
            load_kwargs["local_files_only"] = True
        if self.model_cache_dir:
            load_kwargs["cache_dir"] = self.model_cache_dir

        if self.use_gpu and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # CPU fallback: bfloat16 to save 50% RAM (14 GB instead of 28 GB)
            load_kwargs["device_map"] = None
            load_kwargs["torch_dtype"] = torch.bfloat16

        processor_kwargs = {
            "trust_remote_code": True,
        }
        if local_files_only:
            processor_kwargs["local_files_only"] = True
        if self.model_cache_dir:
            processor_kwargs["cache_dir"] = self.model_cache_dir

        self._processor = AutoProcessor.from_pretrained(
            actual_model_id,
            **processor_kwargs,
        )
        self._model = ModelClass.from_pretrained(
            actual_model_id,
            **load_kwargs,
        )

        if not self.use_gpu or not torch.cuda.is_available():
            self._model = self._model.to("cpu")

        self._model.eval()

        # Log device mapping
        device_info = "cpu"
        try:
            device_info = next(self._model.parameters()).device
        except Exception:
            pass

        logger.info(
            "olmOCR Engine ready in %.2f s (device=%s)",
            time.perf_counter() - t0,
            device_info,
        )
        self._initialized = True

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._initialized = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("olmOCR Engine cleaned up.")

    # ── Image Preprocessing ───────────────────────────────────────────────

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize the longest dimension of the image to exactly 1288px as recommended by AllenAI."""
        image = super().preprocess_image(image)

        # olmOCR expects longest side to be exactly 1288px to fit its VL resolution window
        target_dim = 1288
        w, h = image.size
        longest_side = max(w, h)

        if longest_side != target_dim:
            scale = target_dim / longest_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.debug("Rescaling page image from %dx%d to %dx%d (longest side=%d)", w, h, new_w, new_h, target_dim)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return image

    # ── Document Anchoring ────────────────────────────────────────────────

    def get_anchor_text(self, pdf_path: str, page_number: int) -> str:
        """Extract words and visual normalized coordinates [x0, y0, x1, y1] from PDF metadata.

        Format: 'TEXT [x0,y0,x1,y1]'
        """
        import fitz

        if not os.path.exists(pdf_path):
            logger.warning("PDF path does not exist for anchor extraction: %s", pdf_path)
            return ""

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            page_rect = page.rect
            blocks = page.get_text("blocks", sort=True)  # Sort in reading order
            doc.close()

            anchor_lines = []
            total_chars = 0
            # Target cap around ~4000 characters to prevent prompt bloat
            char_budget = 4000

            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                text = text.strip()
                if not text:
                    continue

                # Normalize to [0, 1] bounds
                nx0 = round(x0 / page_rect.width, 3)
                ny0 = round(y0 / page_rect.height, 3)
                nx1 = round(x1 / page_rect.width, 3)
                ny1 = round(y1 / page_rect.height, 3)

                line = f"{text} [{nx0},{ny0},{nx1},{ny1}]"
                anchor_lines.append(line)
                total_chars += len(line)
                if total_chars >= char_budget:
                    break

            return "\n".join(anchor_lines)

        except Exception as e:
            logger.error("Failed to extract anchor text from PDF: %s", e)
            return ""

    # ── Inference and Extraction ──────────────────────────────────────────

    def extract_text(self, image: Image.Image) -> str:
        """Return raw Markdown transcription (flat pass)."""
        if not self._initialized:
            self.initialize()
        if self.provider == "local":
            return self._run_vlm(image, anchor_text="")
        else:
            rgb_image = self.preprocess_image(image)
            buffered = io.BytesIO()
            rgb_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return self._run_vlm_api(image_base64, anchor_text="")

    def extract_page(
        self,
        image: Image.Image,
        page_number: int = 1,
        pdf_path: Optional[str] = None,
    ) -> LayoutPage:
        """Full layout reconstruction.

        Extracts the page text via olmOCR VLM model (with PDF text anchors)
        and parses it dynamically into layout blocks and structured tables.
        """
        if not self._initialized:
            self.initialize()

        page_w, page_h = image.size

        # Extract anchor text if PDF is provided
        anchor_text = ""
        if pdf_path:
            anchor_text = self.get_anchor_text(pdf_path, page_number)
            logger.debug("[olmOCR] Extracted %d chars of anchor text", len(anchor_text))

        # Run VLM inference based on provider
        if self.provider == "local":
            vlm_markdown = self._run_vlm(image, anchor_text)
        else:
            # For API modes, render/scale image and get base64 encoded PNG
            rgb_image = self.preprocess_image(image)
            buffered = io.BytesIO()
            rgb_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            vlm_markdown = self._run_vlm_api(image_base64, anchor_text)

        logger.debug("[olmOCR] VLM output — %d chars", len(vlm_markdown))

        # Parse layout blocks and tables
        blocks, tables = self._build_layout(
            markdown_text=vlm_markdown,
            page_number=page_number,
        )

        return LayoutPage(
            page_number=page_number,
            width_px=page_w,
            height_px=page_h,
            dpi=0,
            blocks=blocks,
            tables=tables,
        )

    # ── VLM Core Call ─────────────────────────────────────────────────────

    def _run_vlm_api(self, image_base64: str, anchor_text: str) -> str:
        """Call OpenAI API to perform vision-language OCR using Document Anchoring."""
        import httpx
        import json

        system_prompt = OLMOCR_SYSTEM_PROMPT
        user_prompt = build_prompt(anchor_text)

        api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API Key is missing. Please set OPENAI_API_KEY in your .env file or configuration.")
        
        model = self.api_model or "gpt-4o"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        
        logger.info("[olmOCR] Invoking OpenAI VLM API (%s)...", model)
        t0 = time.perf_counter()
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180.0,
            )
            response.raise_for_status()
            response_json = response.json()
            raw_text = response_json["choices"][0]["message"]["content"].strip()
            logger.debug("[olmOCR] OpenAI API completed in %.2f s", time.perf_counter() - t0)
        except Exception as e:
            logger.error("[olmOCR] OpenAI API call failed: %s", e)
            raise

        # Parse structured JSON output
        # Strip markdown fences if VLM wrapped output anyway
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            result = json.loads(raw_text)
            return result.get("natural_text", "")
        except json.JSONDecodeError:
            # Fallback: treat entire response as natural_text
            logger.warning("[olmOCR] Failed to parse VLM response as JSON. Treating as raw Markdown fallback.")
            return raw_text

    def _run_vlm(self, image: Image.Image, anchor_text: str) -> str:
        import torch

        rgb_image = self.preprocess_image(image)

        # Build prompt: anchors + instructions
        if anchor_text.strip():
            user_prompt = (
                "Below is a rendered image of a document page, along with pre-extracted "
                "anchor text and layout coordinates from the PDF.\n\n"
                f"<anchor_text>\n{anchor_text}\n</anchor_text>\n\n"
                "Using the image for layout understanding and the anchor text for character "
                "accuracy, extract all content in natural reading order as structured Markdown."
            )
        else:
            user_prompt = (
                "Below is a rendered image of a scanned document page. "
                "There is no embedded text available (pure scan). "
                "Extract all visible text in natural reading order as structured Markdown."
            )

        # Setup conversation template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback template formatting in case chat template is missing
            prompt_text = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = self._processor(
            text=[prompt_text],
            images=[rgb_image],
            padding=True,
            return_tensors="pt",
        )

        # Move to correct device
        device = next(self._model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        decoded = self._processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return decoded.strip()

    # ── Markdown Block Layout Parsing ─────────────────────────────────────

    def _build_layout(
        self,
        markdown_text: str,
        page_number: int,
    ) -> Tuple[List[LayoutBlock], List[LayoutTable]]:
        """Parse raw VLM Markdown and HTML tables into visual LayoutBlocks."""
        # Normalize carriage returns
        text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = re.split(r"\n{2,}", text)

        blocks: List[LayoutBlock] = []
        tables: List[LayoutTable] = []
        reading_order = 0
        block_idx = 0

        # Check for HTML table tags (olmOCR uses HTML tables for tabular structures)
        html_table_pattern = re.compile(r"(<table\b[^>]*>.*?</table>)", re.DOTALL)
        
        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue

            block_id = f"p{page_number}_b{block_idx:03d}"
            block_idx += 1
            reading_order += 1

            # Case 1: HTML table matches
            html_table_match = html_table_pattern.search(stripped)
            if html_table_match:
                table_html = html_table_match.group(1)
                tbl = _html_table_to_layout_table(
                    table_id=f"p{page_number}_t{len(tables):02d}",
                    html=table_html,
                )
                tables.append(tbl)
                blocks.append(
                    LayoutBlock(
                        block_id=block_id,
                        type="table",
                        text=stripped,
                        html=table_html,
                        reading_order=reading_order,
                        language=_detect_language(stripped),
                        confidence=0.95,
                    )
                )
                continue

            # Case 2: Markdown Table matches
            if _is_markdown_table(stripped):
                tbl = _markdown_to_layout_table(
                    table_id=f"p{page_number}_t{len(tables):02d}",
                    md=stripped,
                )
                tables.append(tbl)
                html = _markdown_table_to_html(stripped)
                blocks.append(
                    LayoutBlock(
                        block_id=block_id,
                        type="table",
                        text=stripped,
                        html=html,
                        reading_order=reading_order,
                        language=_detect_language(stripped),
                        confidence=0.95,
                    )
                )
                continue

            # Case 3: Standard Heading vs Paragraph Text
            block_type: BlockType = _classify_block_type(stripped)
            blocks.append(
                LayoutBlock(
                    block_id=block_id,
                    type=block_type,
                    text=stripped,
                    reading_order=reading_order,
                    language=_detect_language(stripped),
                    confidence=0.90,
                )
            )

        return blocks, tables


# ── Internal Helpers ─────────────────────────────────────────────────────

def _is_markdown_table(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    table_lines = [l for l in lines if _TABLE_ROW_RE.match(l)]
    return len(table_lines) / len(lines) >= 0.75


def _classify_block_type(text: str) -> BlockType:
    if text.startswith("#"):
        # Explicit Markdown headings
        if text.startswith("# "):
            return "title"
        return "heading"
    
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return "paragraph"
    if len(lines) == 1 and len(lines[0]) <= 120 and _HEADING_RE.match(lines[0]):
        return "heading"
    return "paragraph"


def _detect_language(text: str) -> LanguageCode:
    has_tamil = bool(_TAMIL_RE.search(text))
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    has_english = total_letters > 0 and ascii_letters / total_letters > 0.15

    if has_tamil and has_english:
        return "mixed"
    if has_tamil:
        return "ta"
    if has_english:
        return "en"
    return "unknown"


def _markdown_to_layout_table(table_id: str, md: str) -> LayoutTable:
    rows: List[TableRow] = []
    for row_idx, line in enumerate(md.splitlines()):
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-\|]+\|$", line):
            continue
        cells_text = [c.strip() for c in line.strip("|").split("|")]
        cells = [
            TableCell(text=ct, row=row_idx, col=col_idx)
            for col_idx, ct in enumerate(cells_text)
        ]
        rows.append(TableRow(cells=cells))
    return LayoutTable(table_id=table_id, rows=rows)


def _markdown_table_to_html(md: str) -> str:
    lines = [l.strip() for l in md.splitlines() if l.strip()]
    if not lines:
        return ""
    html_rows: List[str] = []
    header_done = False
    for line in lines:
        if re.match(r"^\|[\s\-\|]+\|$", line):
            header_done = True
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not header_done:
            cell_html = "".join(f"<th>{c}</th>" for c in cells)
            html_rows.append(f"<tr>{cell_html}</tr>")
        else:
            cell_html = "".join(f"<td>{c}</td>" for c in cells)
            html_rows.append(f"<tr>{cell_html}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def _html_table_to_layout_table(table_id: str, html: str) -> LayoutTable:
    """Parse visual cells out of a minimal <table> structure."""
    rows: List[TableRow] = []
    tr_blocks = re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, re.DOTALL)
    for row_idx, tr_content in enumerate(tr_blocks):
        cell_blocks = re.findall(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", tr_content, re.DOTALL)
        cells = [
            TableCell(text=c_text.strip(), row=row_idx, col=col_idx)
            for col_idx, c_text in enumerate(cell_blocks)
        ]
        rows.append(TableRow(cells=cells))
    return LayoutTable(table_id=table_id, rows=rows)
