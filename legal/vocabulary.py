"""Tamil legal vocabulary layer — Sub-Registrar domain.

Implements §5.6 of the spec. The vocabulary is:

1. **Injected** into the system prompt so the LLM uses department-approved
   Tamil terminology instead of generic translations
   (e.g. "Stamp Duty" → "முத்திரை தீர்வை", not literal word-by-word).
2. **Optionally enforced** by :func:`ensure_tamil_terms` which post-processes
   an LLM response and rewrites any naïve English-only term into the
   expected `Tamil (English)` form.

The dictionary is intentionally flat and conservative. Add only terms the
Sub-Registrar Department itself uses. Do **not** invent translations.
"""
from __future__ import annotations

import re
from typing import Iterable

#: Master vocabulary. Keys are stable English snake_case ids; values pair the
#: official Tamil term with its English label.
TAMIL_LEGAL_VOCAB: dict[str, tuple[str, str]] = {
    # ── Fees / charges ──
    "stamp_duty": ("முத்திரை தீர்வை", "Stamp Duty"),
    "registration_fee": ("பதிவுக் கட்டணம்", "Registration Fee"),
    "court_fee": ("நீதிமன்றக் கட்டணம்", "Court Fee"),
    "surcharge": ("கூடுதல் கட்டணம்", "Surcharge"),
    "penalty": ("அபராதம்", "Penalty"),

    # ── Document / deed types ──
    "sale_deed": ("விற்பனை பத்திரம்", "Sale Deed"),
    "settlement_deed": ("தீர்வு பத்திரம்", "Settlement Deed"),
    "release_deed": ("விடுவிப்பு பத்திரம்", "Release Deed"),
    "gift_deed": ("தான பத்திரம்", "Gift Deed"),
    "mortgage_deed": ("அடமான பத்திரம்", "Mortgage Deed"),
    "mortgage": ("அடமானம்", "Mortgage"),
    "power_of_attorney": ("வக்கீல் பத்திரம்", "Power of Attorney"),
    "lease_deed": ("குத்தகை பத்திரம்", "Lease Deed"),
    "partition_deed": ("பாகப்பிரிவினை பத்திரம்", "Partition Deed"),

    # ── Roles / offices ──
    "sub_registrar": ("துணைப் பதிவாளர்", "Sub-Registrar"),
    "district_registrar": ("மாவட்டப் பதிவாளர்", "District Registrar"),
    "inspector_general": ("தலைமைப் பதிவாளர்", "Inspector General of Registration"),

    # ── Property / valuation ──
    "guideline_value": ("வழிகாட்டி மதிப்பு", "Guideline Value"),
    "market_value": ("சந்தை மதிப்பு", "Market Value"),
    "encumbrance_certificate": ("சுமை சான்றிதழ்", "Encumbrance Certificate"),
    "patta": ("பட்டா", "Patta"),
    "chitta": ("சிட்டா", "Chitta"),
    "survey_number": ("புல எண்", "Survey Number"),

    # ── Acts / statutes ──
    "stamp_act": ("முத்திரை சட்டம்", "Indian Stamp Act"),
    "registration_act": ("பதிவுச் சட்டம்", "Registration Act"),
    "succession_act": ("வாரிசு சட்டம்", "Succession Act"),
    "hindu_succession_act": ("இந்து வாரிசு சட்டம்", "Hindu Succession Act"),
    "muslim_succession": ("இஸ்லாமிய வாரிசு உரிமை", "Muslim Succession Law"),
    "christian_succession": ("கிறிஸ்தவ வாரிசு சட்டம்", "Christian Succession Law"),
    "standing_orders": ("துணைப் பதிவாளர் நிலை ஆணைகள்", "Standing Orders"),
    "audit_handbook": ("தணிக்கை கையேடு", "Audit Handbook"),

    # ── Workflow nouns ──
    "registration": ("பதிவு", "Registration"),
    "execution": ("செயலாக்கம்", "Execution"),
    "consideration": ("உள்ளடக்கம் / பரிவர்த்தனை மதிப்பு", "Consideration"),
    "executant": ("கொடுப்பவர்", "Executant"),
    "claimant": ("பெறுபவர்", "Claimant"),
    "witness": ("சாட்சி", "Witness"),
}


def vocab_lines() -> str:
    """Render the vocabulary as a stable, prompt-friendly bullet list."""
    parts: list[str] = []
    for _, (tamil, english) in TAMIL_LEGAL_VOCAB.items():
        parts.append(f"- {english} → {tamil}")
    return "\n".join(parts)


# ── Post-processing: replace naked English term with bilingual form ────────

def _english_to_tamil_pairs() -> list[tuple[str, str]]:
    """Sorted by descending key length so longer phrases match first."""
    pairs = [
        (english, tamil) for tamil, english in TAMIL_LEGAL_VOCAB.values()
    ]
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return pairs


_PAIRS_CACHE: list[tuple[str, str]] | None = None
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")


def ensure_tamil_terms(text: str, *, only_when_tamil: bool = True) -> str:
    """Replace every naked occurrence of an English legal term with
    ``"<Tamil> (<English>)"`` so the response is bilingual and unambiguous.

    Parameters
    ----------
    text:
        LLM response text.
    only_when_tamil:
        If ``True`` (default) the rewrite is applied only when the response
        already contains Tamil characters. Pure-English answers are left
        alone — they're going to a non-Tamil reader anyway.
    """
    if not text:
        return text
    if only_when_tamil and not _TAMIL_RE.search(text):
        return text

    global _PAIRS_CACHE
    if _PAIRS_CACHE is None:
        _PAIRS_CACHE = _english_to_tamil_pairs()

    out = text
    for english, tamil in _PAIRS_CACHE:
        if not english or not tamil:
            continue
        # Skip when the bilingual form is already there.
        if f"({english})" in out and tamil in out:
            continue
        # Word-boundary, case-insensitive match. We avoid lookbehind on Tamil
        # characters to keep regex compatible everywhere.
        pattern = re.compile(rf"\b{re.escape(english)}\b", re.IGNORECASE)
        out = pattern.sub(f"{tamil} ({english})", out, count=2)
    return out


def is_tamil_query(text: str) -> bool:
    """True when the user query contains Tamil script characters."""
    return bool(text and _TAMIL_RE.search(text))


def known_terms() -> Iterable[str]:
    """Yield English labels in the vocabulary (for query expansion later)."""
    for _, english in TAMIL_LEGAL_VOCAB.values():
        yield english


__all__ = [
    "TAMIL_LEGAL_VOCAB",
    "vocab_lines",
    "ensure_tamil_terms",
    "is_tamil_query",
    "known_terms",
]
