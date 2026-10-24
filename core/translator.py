"""
Per-block language detection and optional translation.

Detects the language of each PageElement and, when enabled, translates
non-target-language blocks using MarianMT (huggingface/transformers).
Falls back gracefully when the transformers package is not installed.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

# Unicode ranges for common Indic scripts
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
_MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")
_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")

_SCRIPT_MAP = {
    "ta": _TAMIL_RE,
    "hi": _DEVANAGARI_RE,
    "te": _TELUGU_RE,
    "ml": _MALAYALAM_RE,
    "kn": _KANNADA_RE,
}


def detect_block_language(text: str) -> str:
    """
    Detect the dominant language of a text block using script analysis.

    Returns an ISO-639-1 code: 'ta', 'hi', 'te', 'ml', 'kn', or 'en'.
    """
    if not text or not text.strip():
        return "en"

    best_lang = "en"
    best_count = 0

    for lang, pattern in _SCRIPT_MAP.items():
        count = len(pattern.findall(text))
        if count > best_count:
            best_count = count
            best_lang = lang

    # Require a minimum presence to claim non-English
    if best_count < 3:
        return "en"

    return best_lang


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

# Cache loaded MarianMT models so we only load once per language pair
_translation_models: dict = {}


def _get_translator(src_lang: str, tgt_lang: str):
    """
    Load (and cache) a MarianMT model for src→tgt translation.

    Returns (tokenizer, model) or (None, None) if unavailable.
    """
    key = f"{src_lang}-{tgt_lang}"
    if key in _translation_models:
        return _translation_models[key]

    try:
        from transformers import MarianMTModel, MarianTokenizer

        # Helsinki-NLP model naming convention
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        logger.info(f"Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _translation_models[key] = (tokenizer, model)
        return tokenizer, model
    except ImportError:
        logger.warning(
            "transformers package not installed — translation disabled. "
            "Install with: pip install transformers sentencepiece"
        )
        _translation_models[key] = (None, None)
        return None, None
    except Exception as e:
        logger.error(f"Failed to load translation model {src_lang}→{tgt_lang}: {e}")
        _translation_models[key] = (None, None)
        return None, None


def translate_text(text: str, src_lang: str, tgt_lang: str = "en") -> Optional[str]:
    """
    Translate *text* from *src_lang* to *tgt_lang* using MarianMT.

    Returns translated text, or None if translation is unavailable.
    """
    if not text or not text.strip():
        return text

    tokenizer, model = _get_translator(src_lang, tgt_lang)
    if tokenizer is None or model is None:
        return None

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Translation failed ({src_lang}→{tgt_lang}): {e}")
        return None


# ---------------------------------------------------------------------------
# Batch helper for PageElement lists
# ---------------------------------------------------------------------------

def detect_and_translate_elements(
    elements: list,
    enable_translation: bool = False,
    target_lang: str = "en",
) -> list:
    """
    For each PageElement:
      1. Detect block language  →  element.language
      2. If translation is enabled and language != target, translate
         and store the translated text in element.text (original kept
         as a prefix like "[ta] original…\\n[en] translation…").

    Modifies elements in place and returns the same list.
    """
    for el in elements:
        # Skip figures / empty
        if not el.text or el.type == "figure":
            continue

        lang = detect_block_language(el.text)
        el.language = lang

        if enable_translation and lang != target_lang and lang != "en":
            translated = translate_text(el.text, src_lang=lang, tgt_lang=target_lang)
            if translated:
                el.text = f"{el.text}\n\n[{target_lang}] {translated}"

    return elements
