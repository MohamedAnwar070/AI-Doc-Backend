"""PaddleOCR-VL-1.5 Fine-tuned engine — loads base model + LoRA adapter.

This engine extends the base :class:`PaddleVLEngine` with fine-tuned LoRA
weights for improved Tamil + English bilingual document OCR.

Configuration (.env):
    OCR_ENGINE=paddle_vl_ft
    PADDLE_VL_LORA_PATH=D:\\ai-training\\paddle-output-model\\lora-adapter
    # Or on Linux:
    # PADDLE_VL_LORA_PATH=/home/anwar/ai-training/paddle-output-model/lora-adapter

The engine loads the base PaddleOCR-VL-1.5 model, merges the LoRA adapter
on top, and produces inference. Once merged, the adapter adds zero overhead
to inference latency.

GPU / memory notes (same as base PaddleVLEngine):
    bfloat16       ~4 GB    (0.3B model is small)
    float16        ~3 GB
    8-bit quant    ~2 GB
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from PIL import Image

from .paddle_vl_engine import PaddleVLEngine

logger = logging.getLogger(__name__)


class PaddleVLFinetunedEngine(PaddleVLEngine):
    """PaddleOCR-VL-1.5 with fine-tuned LoRA adapter for Tamil/English docs.

    Parameters
    ----------
    lora_adapter_path : str
        Path to the saved LoRA adapter directory. Must contain
        ``adapter_config.json`` and ``adapter_model.safetensors``.
    use_gpu : bool
        Must be True for practical throughput.
    load_in_8bit, load_in_4bit : bool
        Quantisation options (saves VRAM).
    max_new_tokens : int
        Token budget per VLM call.
    model_cache_dir : str, optional
        Cache dir for base model weights.
    """

    def __init__(
        self,
        lora_adapter_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Resolve LoRA path from env or argument
        raw_path = (
            lora_adapter_path
            or os.environ.get("PADDLE_VL_LORA_PATH", "")
        )

        self.lora_adapter_path = ""
        if raw_path:
            import sys
            norm = raw_path.replace("\\", "/")
            if norm.lower().startswith("d:/ai-training/"):
                relative_part = norm[len("d:/ai-training/"):]
                if sys.platform.startswith("win"):
                    self.lora_adapter_path = raw_path
                else:
                    self.lora_adapter_path = os.path.expanduser(
                        os.path.join("~/anwar/ai-training", relative_part)
                    ).replace("\\", "/")
            else:
                self.lora_adapter_path = raw_path

        if not self.lora_adapter_path:
            logger.warning(
                "PaddleVLFinetunedEngine: No LoRA adapter path configured. "
                "Set PADDLE_VL_LORA_PATH in .env or pass lora_adapter_path. "
                "Falling back to base model inference."
            )

    def initialize(self) -> None:
        """Load base model, then merge LoRA adapter on top."""
        if self._initialized:
            return

        # Step 1: Load the base PaddleOCR-VL-1.5 model
        super().initialize()

        # Step 2: Apply LoRA adapter (if path is valid)
        if self.lora_adapter_path and os.path.isdir(self.lora_adapter_path):
            t0 = time.perf_counter()
            logger.info(
                "Merging LoRA adapter from %s ...", self.lora_adapter_path
            )

            try:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(
                    self._model,
                    self.lora_adapter_path,
                )
                # Merge for zero-overhead inference
                self._model = self._model.merge_and_unload()
                self._model.eval()

                logger.info(
                    "LoRA adapter merged in %.1f s — fine-tuned model ready",
                    time.perf_counter() - t0,
                )

            except ImportError:
                logger.error(
                    "peft is not installed. Install it: pip install peft"
                )
                raise

            except Exception as exc:
                logger.error(
                    "Failed to load LoRA adapter from %s: %s. "
                    "Using base model.",
                    self.lora_adapter_path,
                    exc,
                )
        else:
            if self.lora_adapter_path:
                logger.warning(
                    "LoRA adapter path not found: %s — using base model",
                    self.lora_adapter_path,
                )
