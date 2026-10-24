"""Sub-Registrar legal knowledge package.

Phase 1a: Tamil vocabulary + prompt builder + citation post-processor.
Phase 2 (future): rule_engine, slot_filler, fee_calculator.
"""

from .prompts import (
    SubRegistrarPromptBuilder,
    build_citation_block,
    build_system_prompt,
)
from .vocabulary import TAMIL_LEGAL_VOCAB, vocab_lines, ensure_tamil_terms

__all__ = [
    "TAMIL_LEGAL_VOCAB",
    "SubRegistrarPromptBuilder",
    "build_citation_block",
    "build_system_prompt",
    "ensure_tamil_terms",
    "vocab_lines",
]
