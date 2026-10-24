"""Unit tests for :mod:`core.text_normalizer`."""
from __future__ import annotations

from core.text_normalizer import (
    has_tamil,
    is_predominantly_tamil,
    normalize,
    normalize_lines,
)


# ── has_tamil / is_predominantly_tamil ─────────────────────────────────────

def test_has_tamil_detects_tamil_glyphs() -> None:
    assert has_tamil("வணக்கம்") is True
    assert has_tamil("hello world") is False
    assert has_tamil("") is False


def test_is_predominantly_tamil_threshold() -> None:
    pure_tamil = "வணக்கம் உலகம்"
    mixed = "hello உலகம்"
    pure_english = "hello world"
    assert is_predominantly_tamil(pure_tamil) is True
    assert is_predominantly_tamil(mixed) is True  # >= 20% Tamil letters
    assert is_predominantly_tamil(pure_english) is False
    assert is_predominantly_tamil("") is False


# ── Unicode normalisation ──────────────────────────────────────────────────

def test_normalize_applies_nfc() -> None:
    # 'é' has both pre-composed (U+00E9) and decomposed (e + U+0301) forms.
    # NFC must collapse them to the same string regardless of input form.
    decomposed = "e\u0301"
    precomposed = "\u00e9"
    assert decomposed != precomposed
    assert normalize(decomposed) == normalize(precomposed) == "é"


def test_normalize_strips_zero_width_chars() -> None:
    out = normalize("hello\u200Bworld\u200C\u200Dtext\uFEFF")
    assert out == "helloworldtext"


# ── Whitespace / newlines ──────────────────────────────────────────────────

def test_normalize_collapses_multiple_spaces() -> None:
    assert normalize("hello    world\t\tfoo") == "hello world foo"


def test_normalize_collapses_excessive_newlines() -> None:
    text = "para1\n\n\n\npara2"
    assert normalize(text) == "para1\n\npara2"


def test_normalize_strips_trailing_whitespace_before_newline() -> None:
    assert normalize("line1   \nline2") == "line1\nline2"


def test_normalize_returns_empty_for_falsy_input() -> None:
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


# ── Latin digit confusables ────────────────────────────────────────────────

def test_normalize_fixes_O_to_zero_in_dates() -> None:
    # "01.O5.2O25" should become "01.05.2025" — O between digits → 0.
    assert normalize("Date: 01.O5.2O25") == "Date: 01.05.2025"


def test_normalize_does_not_corrupt_words() -> None:
    # "Hello" must NOT have its O converted to 0 because the token isn't numeric.
    assert "Hello" in normalize("Hello world")


def test_normalize_l_to_one_in_numeric_run() -> None:
    # The conservative regex only rewrites tokens that *start* with a digit
    # (so we don't corrupt variable names or English words). "1l5" → "115".
    assert normalize("page 1l5") == "page 115"


def test_normalize_does_not_corrupt_letter_prefixed_tokens() -> None:
    # "AB-l23" must NOT become "AB-123" — leading letters mean it's likely
    # an identifier, not a number.
    assert normalize("ref AB-l23") == "ref AB-l23"


# ── Tamil digit confusables ────────────────────────────────────────────────

def test_normalize_converts_tamil_digits_inside_numeric_runs() -> None:
    # ௦௧.௦௫.௨௦௨௫ → 01.05.2025  (purely Tamil digits with separators).
    out = normalize("date ௦௧.௦௫.௨௦௨௫")
    assert "01.05.2025" in out


def test_normalize_keeps_isolated_tamil_digits_in_text() -> None:
    # A lone Tamil digit inside Tamil text must stay.
    assert "௧" in normalize("அத்தியாயம் ௧ தொடக்கம்")


# ── Idempotency ────────────────────────────────────────────────────────────

def test_normalize_is_idempotent() -> None:
    text = "Some\u200Btext  with\nweird\r\nformatting and  01.O5.2O25"
    once = normalize(text)
    twice = normalize(once)
    assert once == twice


# ── Bulk helpers ───────────────────────────────────────────────────────────

def test_normalize_lines_drops_empties() -> None:
    out = normalize_lines(["", "  ", "hello", "world\u200B"])
    assert out == ["hello", "world"]


# ── Watermarks & Noise Stripping ───────────────────────────────────────────

def test_normalize_strips_scanner_watermarks() -> None:
    text = (
        "DEED OF SALE\n"
        "This Deed of Sale is executed on 10th May 2024.\n"
        "Scanned by CamScanner\n"
        "www.camscanner.com"
    )
    normalized = normalize(text)
    assert "DEED OF SALE" in normalized
    assert "10th May 2024" in normalized
    assert "CamScanner" not in normalized
    assert "www.camscanner.com" not in normalized


def test_normalize_strips_stray_noise_lines_while_keeping_content() -> None:
    text = (
        "Item No. 1: Bounded North by Road\n"
        "_. _ _\n"
        "...\n"
        "|\n"
        "South by Survey No 102"
    )
    normalized = normalize(text)
    assert "Bounded North by Road" in normalized
    assert "South by Survey No 102" in normalized
    assert "_" not in normalized # Stray underlines should be removed
    assert "|" not in normalized # Stray border symbol should be removed


def test_normalize_optional_strip_noise_disable() -> None:
    text = "Valid text\nScanned by CamScanner"
    assert "CamScanner" in normalize(text, strip_noise=False)

