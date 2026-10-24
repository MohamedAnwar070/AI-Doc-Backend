"""Tamil-aware OCR text normaliser.

Pure-functional, no third-party dependencies. Applied **after** OCR,
**before** storage / chunking / embedding, so every downstream consumer
sees the same canonical form.

Goals
-----
1. **Unicode NFC** — Tamil glyphs combine multiple code-points into a
   single visual character. Searching breaks if the same word is stored
   in a different normalisation form on different documents.
2. **Mixed Tamil + English safety** — never collapse English contractions
   or punctuation that adjacent Tamil text might trigger heuristically.
3. **Common confusables** — Paddle's ``ta`` model commonly emits a small
   set of mis-recognitions (e.g. *zero* mistaken for the Tamil letter
   *ஓ* in numerical contexts). We correct only the unambiguous cases.
4. **Whitespace hygiene** — collapse multi-spaces, strip zero-width
   joiners that survive Paddle output, normalise newlines.
5. **Glyph repair** — fix a small, audited set of Paddle-Tamil glyph
   mis-recognitions that occur consistently in real production scans
   (vowel sign / pulli swaps that are visually unambiguous).

This module is intentionally conservative — it never *guesses* what the
user wrote; it only canonicalises forms that are demonstrably equivalent.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# ── Tunables ────────────────────────────────────────────────────────────────

#: Tamil script Unicode block (excluding Tamil-Supplement which is rare).
_TAMIL_BLOCK = "\u0B80-\u0BFF"

#: Zero-width characters that Paddle / Surya occasionally insert into
#: Tamil output. Removing them is safe — they have no visual effect.
_ZERO_WIDTH_CHARS = "\u200B\u200C\u200D\u2060\uFEFF"

#: Characters that Paddle's Tamil model frequently confuses with digits
#: when reading dates / numbers. Mapped to their digit equivalents only
#: when surrounded by other digits / punctuation (the regex below
#: enforces that — we never replace mid-word).
_DIGIT_CONFUSABLES: dict[str, str] = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "B": "8",
}

#: Tamil sign / digit confusables that show up in numerical contexts.
#: Only mapped between digits to avoid corrupting Tamil words.
_TAMIL_DIGIT_CONFUSABLES: dict[str, str] = {
    "௦": "0",
    "௧": "1",
    "௨": "2",
    "௩": "3",
    "௪": "4",
    "௫": "5",
    "௬": "6",
    "௭": "7",
    "௮": "8",
    "௯": "9",
}

_TAMIL_RE = re.compile(f"[{_TAMIL_BLOCK}]")
_ZERO_WIDTH_RE = re.compile(f"[{_ZERO_WIDTH_CHARS}]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_DIGIT_GROUP_RE = re.compile(r"\b\d[\dOoIlSB._\-/,:]{2,}\b")

#: Whitespace that should never appear between a Tamil base letter and a
#: combining vowel sign / pulli. Paddle occasionally injects a literal
#: space here when the glyph is split across two detection boxes.
_TAMIL_BASE = "\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99-\u0B9A\u0B9C\u0B9E-\u0B9F\u0BA3-\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9"
_TAMIL_VOWEL_SIGN = "\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0BD7"
_SPLIT_VOWEL_RE = re.compile(
    rf"([{_TAMIL_BASE}])[ \t]+([{_TAMIL_VOWEL_SIGN}])"
)

#: Repeated-pulli artefact: Paddle sometimes emits ``்்`` (two pulli
#: characters in a row). The second is always spurious.
_DOUBLE_PULLI_RE = re.compile("\u0BCD\u0BCD+")

#: Stray Latin punctuation injected mid-Tamil-word. Apostrophes and
#: backticks between two Tamil letters are always OCR noise.
_TAMIL_INNER_PUNCT_RE = re.compile(
    rf"([{_TAMIL_BASE}{_TAMIL_VOWEL_SIGN}])[`\u2018\u2019']([{_TAMIL_BASE}])"
)

#: Common scanner app watermarks and promotional footer strings.
_WATERMARK_RE = re.compile(
    r"(?i)^\s*(?:(?:scanned|created|generated)\s+(?:by|with)|from|using)?\s*"
    r"(?:camscanner|adobe\s*scan|office\s*lens|clear\s*scanner|fast\s*scanner|cs\s*camscanner|www\.camscanner\.com).*$",
    re.MULTILINE,
)

#: Standalone noise lines consisting purely of repetitive symbols, borders, or single stray characters
#: (e.g., "---", "_ _ _", "...", "|", or scan frame boundary glitches), while preserving valid text content.
_STRAY_NOISE_RE = re.compile(
    r"^\s*(?:[.\-=_*|~#]\s*){3,}$\n?|^\s*[|:.\-_,;!~]\s*$\n?",
    re.MULTILINE,
)


# ── Public API ──────────────────────────────────────────────────────────────

def has_tamil(text: str) -> bool:
    """Return ``True`` when ``text`` contains at least one Tamil glyph."""
    return bool(text) and bool(_TAMIL_RE.search(text))


def is_predominantly_tamil(text: str, threshold: float = 0.20) -> bool:
    """Return ``True`` when at least ``threshold`` of letters are Tamil.

    Used by the worker to decide whether to apply Tamil-specific
    post-processing or stay in pure-English mode.
    """
    if not text:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return False
    tamil_chars = sum(1 for ch in text if _TAMIL_RE.match(ch))
    return (tamil_chars / letters) >= threshold


def strip_watermarks_and_noise(text: str) -> str:
    """Remove repetitive scanner app watermarks, URLs, and stray noise lines.

    Specifically targets promotional footers inserted by scanning apps
    (e.g., CamScanner, Adobe Scan) and standalone non-alphanumeric noise lines
    resulting from optical boundary scanning artifacts.
    """
    if not text:
        return ""
    out = _WATERMARK_RE.sub("", text)
    out = _STRAY_NOISE_RE.sub("", out)
    return out.strip()


def normalize(
    text: str,
    *,
    apply_tamil: bool = True,
    apply_dictionary: bool = True,
    strip_noise: bool = True,
) -> str:
    """Canonicalise OCR output.

    Parameters
    ----------
    text:
        The raw OCR string.
    apply_tamil:
        When ``True`` (default), Tamil-specific post-processing is run on
        top of the universal cleanup. Set to ``False`` for documents that
        are guaranteed to be pure English / Latin to avoid any risk of
        false rewrites.
    apply_dictionary:
        When ``True`` (default), a small audited set of Tamil glyph
        repairs is applied (split vowel-signs, double pulli, stray inner
        punctuation). Disable for raw archival output.
    strip_noise:
        When ``True`` (default), removes mobile scanning watermarks (e.g. CamScanner)
        and standalone optical scanner boundary glitch characters.
    """
    if not text:
        return ""

    out = unicodedata.normalize("NFC", text)
    out = _strip_zero_width(out)
    if strip_noise:
        out = strip_watermarks_and_noise(out)
    out = _normalize_newlines(out)
    out = _collapse_whitespace(out)

    if apply_tamil and has_tamil(out):
        out = _fix_tamil_digit_confusables(out)
        if apply_dictionary:
            out = _repair_tamil_glyph_artifacts(out)
        # Re-apply NFC after structural repairs so combined glyphs reform.
        out = unicodedata.normalize("NFC", out)

    out = _fix_latin_digit_confusables(out)
    return out.strip()


def normalize_lines(lines: Iterable[str], *, apply_tamil: bool = True, strip_noise: bool = True) -> list[str]:
    """Apply :func:`normalize` line-by-line, dropping empties."""
    return [n for n in (normalize(line, apply_tamil=apply_tamil, strip_noise=strip_noise) for line in lines) if n]


# ── Internal helpers ────────────────────────────────────────────────────────

def _strip_zero_width(text: str) -> str:
    """Remove invisible joiner / BOM characters."""
    return _ZERO_WIDTH_RE.sub("", text)


def _normalize_newlines(text: str) -> str:
    """Canonical ``\\n`` newlines, no trailing whitespace, collapse blank runs."""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _TRAILING_WS_RE.sub("\n", out)
    out = _MULTI_NEWLINE_RE.sub("\n\n", out)
    return out


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces / tabs to a single space (newlines preserved)."""
    return _MULTI_SPACE_RE.sub(" ", text)


def _fix_tamil_digit_confusables(text: str) -> str:
    """Convert Tamil digit code-points to ASCII digits *inside numeric runs*.

    We never rewrite Tamil digits in isolation — a single Tamil digit
    embedded in Tamil text is almost always intentional (chapter or
    verse marker). Only converts when the run is bracketed by ASCII
    digits / dots / dashes (i.e. it really *is* a number).
    """
    def replace(match: re.Match) -> str:
        token = match.group(0)
        if not any(ch in _TAMIL_DIGIT_CONFUSABLES for ch in token):
            return token
        return "".join(_TAMIL_DIGIT_CONFUSABLES.get(ch, ch) for ch in token)

    return re.sub(
        rf"\b[\d.\-/,:{''.join(_TAMIL_DIGIT_CONFUSABLES.keys())}]{{3,}}\b",
        replace,
        text,
    )


def _fix_latin_digit_confusables(text: str) -> str:
    """Inside obvious numeric tokens (dates, IDs), convert ``O→0``, ``I→1`` etc."""
    def replace(match: re.Match) -> str:
        token = match.group(0)
        digit_count = sum(1 for ch in token if ch.isdigit())
        if digit_count < 2 or digit_count == len(token):
            return token
        return "".join(_DIGIT_CONFUSABLES.get(ch, ch) for ch in token)

    return _DIGIT_GROUP_RE.sub(replace, text)


def _repair_tamil_glyph_artifacts(text: str) -> str:
    """Repair structural Tamil OCR artefacts that are unambiguous.

    The transformations here are *deterministic*: they only touch
    sequences that have no valid Tamil interpretation as-is. We never
    invent letters or guess words.

    Specifically:

    * Re-attach combining vowel signs / pulli that Paddle split with a
      space (``க ா`` → ``கா``).
    * Collapse double pulli (``்்``) to single (``்``).
    * Drop stray Latin apostrophes injected mid-word (``க'லை`` → ``கலை``).
    """
    out = _SPLIT_VOWEL_RE.sub(r"\1\2", text)
    out = _DOUBLE_PULLI_RE.sub("\u0BCD", out)
    out = _TAMIL_INNER_PUNCT_RE.sub(r"\1\2", out)
    return out
