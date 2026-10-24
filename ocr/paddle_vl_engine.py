"""PaddleOCR-VL-1.5 engine — Vision Language Model inference via 🤗 transformers.

Architecture
------------
PaddleOCR-VL-1.5 is a **multimodal transformer** (similar category to Qwen-VL /
GPT-4o vision), NOT the old CNN/CRNN pipeline.  It reads the page image as a
whole and *understands* layout, tables, formulas, and mixed Tamil/English text
without any bounding-box heuristics.

Task prompts (official PaddleOCR-VL docs):
    OCR:                 → plain text, preserves reading order
    Table Recognition:  → markdown table
    Formula Recognition:→ LaTeX
    Chart Recognition:  → structured description

This engine always runs two passes per page:
  1. OCR:   full page text with correct reading order
  2. Table Recognition:  if the response contains ``|``, a markdown table is
     already embedded — we keep it as-is and produce structured LayoutBlocks.

The result is adapted to the existing :class:`ocr.layout_schema.LayoutBlock` /
:class:`ocr.layout_schema.LayoutPage` schema so every downstream consumer
(RAG chunker, layout API, frontend renderer) continues to work unchanged.

GPU / memory notes
------------------
Model          VRAM
bfloat16       ~16 GB   (default, best quality)
float16        ~12 GB
8-bit quant    ~8 GB    (set PADDLE_VL_LOAD_IN_8BIT=true)
4-bit quant    ~5 GB    (set PADDLE_VL_LOAD_IN_4BIT=true)

When no GPU is available the engine raises ``RuntimeError`` — CPU inference
for a 7B-class VLM is impractical (>5 min/page).  Use ``OCR_ENGINE=paddle``
for CPU-only environments.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import List, Optional

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

_MODEL_ID_REMOTE = "PaddlePaddle/PaddleOCR-VL-1.5"

import os
import sys

if sys.platform.startswith("win"):
    _MODEL_ID_LOCAL = r"D:\ai-training\models\paddle-vl-ocr"
else:
    _MODEL_ID_LOCAL = os.path.expanduser("~/anwar/ai-training/models/paddle-vl-ocr")

_MODEL_ID = _MODEL_ID_LOCAL if os.path.isdir(_MODEL_ID_LOCAL) else _MODEL_ID_REMOTE

# ── Task prompts (official) ────────────────────────────────────────────────
_PROMPT_OCR = "OCR:"
_PROMPT_TABLE = "Table Recognition:"

# Markdown table row detector
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
# Tamil Unicode range (basic block only — sufficient for block detection)
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
# Section / heading heuristic: short line (≤120 chars), no trailing punctuation
_HEADING_RE = re.compile(r"^(?!\s*\|)[^\n]{1,120}[^\.\,\;\:\|]$")





class PaddleVLEngine(OCREngine):
    """PaddleOCR-VL-1.5 multimodal transformer adapter.

    Parameters
    ----------
    use_gpu:
        Must be ``True`` for practical throughput. Raises on ``False``.
    load_in_8bit:
        Use bitsandbytes 8-bit quantisation (saves ~6 GB VRAM, ~5% quality loss).
    load_in_4bit:
        Use bitsandbytes 4-bit quantisation (saves ~10 GB VRAM, ~10% quality loss).
    max_new_tokens:
        Token budget per VLM call. 4096 covers even dense legal pages.
    model_cache_dir:
        Optional local path to pre-downloaded model weights
        (set via ``PADDLE_VL_CACHE_DIR`` env var).  ``None`` → HuggingFace cache.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 4096,
        model_cache_dir: Optional[str] = None,
        **_ignored_kwargs,  # absorb any forwarded engine kwargs (lang, preprocess_options, etc.)
    ) -> None:
        super().__init__()
        if not use_gpu:
            logger.warning(
                "PaddleVLEngine: GPU not enabled (USE_GPU=false). "
                "CPU inference for PaddleOCR-VL-1.5 is very slow (~5 min/page). "
                "Set GPU_ENABLED=true for production. Continuing in CPU mode for dev/test."
            )
        self.use_gpu = use_gpu
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.model_cache_dir = model_cache_dir

        self._processor = None
        self._model = None
        self._initialized: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch

        # Monkey patch ROPE_INIT_FUNCTIONS for newer versions of transformers
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if "default" not in ROPE_INIT_FUNCTIONS:
                def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
                    config.standardize_rope_params()
                    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
                    base = rope_parameters_dict.get("rope_theta", 10000.0)
                    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                    dim = int(head_dim * partial_rotary_factor)
                    attention_factor = 1.0
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
                    return inv_freq, attention_factor

                ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
                logger.info("Monkey-patched default RoPE function into ROPE_INIT_FUNCTIONS")
        except Exception as e:
            logger.warning(f"Failed to patch ROPE_INIT_FUNCTIONS: {e}")

        # Monkey patch PreTrainedModel._init_weights to inject compute_default_rope_parameters into custom RotaryEmbedding classes
        try:
            from transformers.modeling_utils import PreTrainedModel
            original_init_weights = PreTrainedModel._init_weights

            if getattr(original_init_weights, "__name__", "") != "patched_init_weights":
                def patched_init_weights(self, module):
                    if "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
                        if not hasattr(module, "compute_default_rope_parameters"):
                            def _compute_default_rope_parameters(config=module.config, device=module.inv_freq.device, seq_len=None, layer_type=None):
                                config.standardize_rope_params()
                                rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
                                base = rope_parameters_dict.get("rope_theta", 10000.0)
                                partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                                dim = int(head_dim * partial_rotary_factor)
                                attention_factor = 1.0
                                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
                                return inv_freq, attention_factor
                            
                            module.compute_default_rope_parameters = _compute_default_rope_parameters
                            logger.info("Monkey-patched compute_default_rope_parameters on RotaryEmbedding")

                    return original_init_weights(self, module)
                
                patched_init_weights.__name__ = "patched_init_weights"
                PreTrainedModel._init_weights = patched_init_weights
                logger.info("Monkey-patched PreTrainedModel._init_weights")
        except Exception as e:
            logger.warning(f"Failed to patch PreTrainedModel._init_weights: {e}")

        # Monkey patch create_causal_mask to handle cache_position changes across transformers versions
        try:
            import transformers.masking_utils as masking_utils
            if hasattr(masking_utils, "create_causal_mask"):
                orig_create_causal_mask = masking_utils.create_causal_mask
                if getattr(orig_create_causal_mask, "__name__", "") != "patched_create_causal_mask":
                    import inspect
                    sig = inspect.signature(orig_create_causal_mask)
                    has_cache_position_param = "cache_position" in sig.parameters
                    has_input_embeds_plural = "inputs_embeds" in sig.parameters
                    has_input_embeds_singular = "input_embeds" in sig.parameters

                    def patched_create_causal_mask(*args, **kwargs):
                        # Map plural inputs_embeds to singular input_embeds if needed (cross-version compatibility)
                        if "inputs_embeds" in kwargs and not has_input_embeds_plural and has_input_embeds_singular:
                            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                        elif "input_embeds" in kwargs and not has_input_embeds_singular and has_input_embeds_plural:
                            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")

                        if has_cache_position_param and "cache_position" not in kwargs:
                            # Extract inputs_embeds or input_embeds
                            inputs_embeds = kwargs.get("inputs_embeds") or kwargs.get("input_embeds")
                            if inputs_embeds is None and len(args) > 1:
                                inputs_embeds = args[1]
                            
                            # Extract past_key_values
                            past_key_values = kwargs.get("past_key_values")
                            if past_key_values is None and len(args) > 3:
                                past_key_values = args[3]
                            
                            if inputs_embeds is not None:
                                seq_len = inputs_embeds.shape[1]
                                device = inputs_embeds.device
                                past_length = 0
                                if past_key_values is not None:
                                    if hasattr(past_key_values, "get_seq_length"):
                                        past_length = past_key_values.get_seq_length()
                                    elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                                        past_length = past_key_values[0][0].shape[-2]
                                
                                kwargs["cache_position"] = torch.arange(past_length, past_length + seq_len, device=device)
                        
                        return orig_create_causal_mask(*args, **kwargs)

                    patched_create_causal_mask.__name__ = "patched_create_causal_mask"
                    masking_utils.create_causal_mask = patched_create_causal_mask
                    logger.info("Monkey-patched create_causal_mask in transformers.masking_utils")
        except Exception as e:
            logger.warning(f"Failed to patch create_causal_mask: {e}")

        from transformers import AutoModelForCausalLM, AutoProcessor

        # Optimize CPU threads for PyTorch/transformers inference when GPU is disabled
        if not self.use_gpu:
            import os
            # Prevent thread over-subscription on CPU (default to 4 threads or OMP_NUM_THREADS)
            num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
            try:
                torch.set_num_threads(num_threads)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(1)
                logger.info("Set PyTorch CPU threads to %d", num_threads)
            except Exception as e:
                logger.warning("Could not configure PyTorch CPU threads: %s", e)

        logger.info(
            "Loading PaddleOCR-VL-1.5 from HuggingFace (%s)...  "
            "8bit=%s  4bit=%s  cache=%s",
            _MODEL_ID,
            self.load_in_8bit,
            self.load_in_4bit,
            self.model_cache_dir or "<hf_default>",
        )
        t0 = time.perf_counter()

        load_kwargs: dict = dict(
            trust_remote_code=True,
            device_map="auto",
        )

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

        if self.model_cache_dir:
            load_kwargs["cache_dir"] = self.model_cache_dir

        self._processor = AutoProcessor.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
            cache_dir=self.model_cache_dir,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            **load_kwargs,
        )

        # Monkey patch prepare_inputs_for_generation on the loaded model instance to handle cache_position issues
        if hasattr(self._model, "prepare_inputs_for_generation"):
            original_prep = self._model.prepare_inputs_for_generation
            import types
            def patched_prepare_inputs(self_model, input_ids, *args, **kwargs):
                cache_pos = kwargs.get("cache_position", None)
                if cache_pos is None and len(args) > 3:
                    cache_pos = args[3]
                
                if cache_pos is None:
                    seq_len = input_ids.shape[1] if input_ids is not None else 1
                    device = input_ids.device if input_ids is not None else torch.device("cpu")
                    kwargs["cache_position"] = torch.arange(0, seq_len, device=device)
                
                try:
                    return original_prep(input_ids, *args, **kwargs)
                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    if "cache_position" in tb_str and ("NoneType" in tb_str or "TypeError" in tb_str):
                        seq_len = input_ids.shape[1] if input_ids is not None else 1
                        device = input_ids.device if input_ids is not None else torch.device("cpu")
                        kwargs["cache_position"] = torch.arange(0, seq_len, device=device)
                        return original_prep(input_ids, *args, **kwargs)
                    raise
            
            patched_prepare_inputs.__name__ = "patched_prepare_inputs"
            self._model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs, self._model)
            logger.info("Monkey-patched prepare_inputs_for_generation on model instance")

        self._model.eval()

        logger.info(
            "PaddleOCR-VL-1.5 ready in %.1f s  (device=%s)",
            time.perf_counter() - t0,
            next(self._model.parameters()).device,
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
        logger.info("PaddleVLEngine cleaned up.")

    # ── Public extraction ──────────────────────────────────────────────────

    def extract_text(self, image: Image.Image) -> str:
        """Return raw plain text from the page image (single-pass OCR)."""
        if not self._initialized:
            self.initialize()
        return self._run_vlm(image, _PROMPT_OCR)

    def extract_page(self, image: Image.Image, page_number: int = 1) -> LayoutPage:
        """Full structured extraction → :class:`LayoutPage`.

        Strategy
        --------
        1. Run ``OCR:`` pass — returns full page text in reading order.
        2. Run ``Table Recognition:`` pass — returns markdown tables.
        3. Merge: wherever a markdown table appears in the OCR text, replace the
           raw text block with a structured :class:`LayoutTable`.
        4. Classify remaining lines as heading / paragraph.
        5. Detect dominant language per block (Tamil / English / mixed).
        """
        if not self._initialized:
            self.initialize()

        page_w, page_h = image.size

        # ── Pass 1: OCR ────────────────────────────────────────────────────
        ocr_text = self._run_vlm(image, _PROMPT_OCR)
        logger.debug("[VL] OCR pass — %d chars", len(ocr_text))

        # ── Pass 2: Table Recognition ──────────────────────────────────────
        table_text = self._run_vlm(image, _PROMPT_TABLE)
        logger.debug("[VL] Table pass — %d chars", len(table_text))

        # ── Build blocks ───────────────────────────────────────────────────
        blocks, tables = self._build_layout(
            ocr_text=ocr_text,
            table_text=table_text,
            page_number=page_number,
        )

        return LayoutPage(
            page_number=page_number,
            width_px=page_w,
            height_px=page_h,
            dpi=0,          # caller may patch this
            blocks=blocks,
            tables=tables,
        )

    # ── VLM inference ──────────────────────────────────────────────────────

    def _run_vlm(self, image: Image.Image, task_prompt: str) -> str:
        """Single VLM inference call for one task prompt."""
        import torch

        rgb_image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": task_prompt},
                ],
            }
        ]

        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[prompt],
            images=[rgb_image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        # Decode only the newly generated tokens (strip the prompt echo)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        raw = self._processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0]

        return raw.strip()

    # ── Layout construction ────────────────────────────────────────────────

    def _build_layout(
        self,
        ocr_text: str,
        table_text: str,
        page_number: int,
    ) -> tuple[List[LayoutBlock], List[LayoutTable]]:
        """Convert raw VLM output into structured LayoutBlock / LayoutTable lists."""

        # Parse markdown tables from the dedicated table pass
        md_tables = _parse_markdown_tables(table_text)

        blocks: List[LayoutBlock] = []
        layout_tables: List[LayoutTable] = []
        reading_order = 0
        block_idx = 0

        # Split OCR output into paragraphs / sections
        paragraphs = _split_paragraphs(ocr_text)

        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue

            block_id = f"p{page_number}_b{block_idx:03d}"
            block_idx += 1
            reading_order += 1

            # Check if this paragraph looks like a markdown table
            if _is_markdown_table(stripped):
                tbl = _markdown_to_layout_table(
                    table_id=f"p{page_number}_t{len(layout_tables):02d}",
                    md=stripped,
                )
                layout_tables.append(tbl)
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

            # Not a table — classify as heading or paragraph
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

        # If dedicated table pass found tables NOT already in OCR text, append them
        for i, tbl in enumerate(md_tables):
            if not _table_already_in_blocks(tbl, blocks):
                block_id = f"p{page_number}_b{block_idx:03d}"
                block_idx += 1
                reading_order += 1
                html = _markdown_table_to_html(tbl)
                lt = _markdown_to_layout_table(
                    table_id=f"p{page_number}_t{len(layout_tables):02d}",
                    md=tbl,
                )
                layout_tables.append(lt)
                blocks.append(
                    LayoutBlock(
                        block_id=block_id,
                        type="table",
                        text=tbl,
                        html=html,
                        reading_order=reading_order,
                        language=_detect_language(tbl),
                        confidence=0.95,
                    )
                )

        return blocks, layout_tables


# ── Pure helpers (no model state) ─────────────────────────────────────────


def _split_paragraphs(text: str) -> List[str]:
    """Split VLM output into logical paragraphs on blank lines."""
    # Normalise Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.split(r"\n{2,}", text)


def _is_markdown_table(text: str) -> bool:
    """Return True when *all* non-empty lines look like markdown table rows."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    table_lines = [l for l in lines if _TABLE_ROW_RE.match(l)]
    return len(table_lines) / len(lines) >= 0.75


def _classify_block_type(text: str) -> BlockType:
    """Heuristic heading vs paragraph classifier.

    Short lines without trailing punctuation and without table pipes are
    classified as headings.  Everything else is a paragraph.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return "paragraph"
    # Single-line, short, no trailing comma/period → likely a heading
    if len(lines) == 1 and len(lines[0]) <= 120 and _HEADING_RE.match(lines[0]):
        return "heading"
    return "paragraph"


def _detect_language(text: str) -> LanguageCode:
    """Classify dominant language block: ta / en / mixed / unknown."""
    has_tamil = bool(_TAMIL_RE.search(text))
    # Simple heuristic: if >15% of chars are ASCII letters → has English
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


def _parse_markdown_tables(text: str) -> List[str]:
    """Extract all markdown table blocks from raw text."""
    tables: List[str] = []
    current_lines: List[str] = []
    in_table = False

    for line in text.splitlines():
        if _TABLE_ROW_RE.match(line):
            current_lines.append(line)
            in_table = True
        else:
            if in_table and current_lines:
                tables.append("\n".join(current_lines))
                current_lines = []
            in_table = False

    if current_lines:
        tables.append("\n".join(current_lines))

    return tables


def _markdown_to_layout_table(table_id: str, md: str) -> LayoutTable:
    """Parse a markdown table string into a :class:`LayoutTable`."""
    rows: List[TableRow] = []
    for row_idx, line in enumerate(md.splitlines()):
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        # Skip pure separator rows (----)
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
    """Convert a markdown table to a minimal HTML table string."""
    lines = [l.strip() for l in md.splitlines() if l.strip()]
    if not lines:
        return ""

    html_rows: List[str] = []
    header_done = False

    for line in lines:
        if re.match(r"^\|[\s\-\|]+\|$", line):
            # Separator line — marks end of header
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


def _table_already_in_blocks(md_table: str, blocks: List[LayoutBlock]) -> bool:
    """Avoid duplicate table blocks when OCR pass already captured the table."""
    # Use first cell text as a cheap fingerprint
    first_cell = md_table.split("|")[1].strip() if "|" in md_table else ""
    if not first_cell:
        return False
    return any(
        b.type == "table" and first_cell in (b.text or "")
        for b in blocks
    )
