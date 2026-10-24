"""Unit tests for the security-critical helpers in :mod:`api.upload`.

These tests pin the contract that any caller-supplied filename is
sanitised into a path-traversal-safe basename before being written
to ``settings.upload_dir``.

The helpers live in :mod:`api._upload_utils` precisely so we can
test them without dragging the heavyweight FastAPI router stack
(PyMuPDF, sklearn, etc.) into a unit test.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Load the helper module directly from disk to avoid triggering
# ``api/__init__.py`` (which eagerly imports every router).
_HELPER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "api" / "_upload_utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "ai_doc_retrieval._upload_utils", _HELPER_PATH
)
assert _spec and _spec.loader, "could not locate api/_upload_utils.py"
_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, _module)
_spec.loader.exec_module(_module)

sanitise_filename = _module.sanitise_filename
validate_language = _module.validate_language


class TestSanitiseFilename:
    def test_strips_directory_segments(self) -> None:
        assert sanitise_filename("../../etc/passwd") == "passwd.pdf"

    def test_strips_absolute_path(self) -> None:
        assert sanitise_filename("/etc/passwd") == "passwd.pdf"

    def test_strips_windows_path(self) -> None:
        # On POSIX, backslash is just a regular char; the regex still
        # collapses it to '_' so we never end up with a separator.
        out = sanitise_filename("C:\\Users\\evil\\notes.pdf")
        assert "/" not in out and "\\" not in out
        assert out.endswith(".pdf")

    def test_keeps_safe_filename(self) -> None:
        assert sanitise_filename("invoice_2026.pdf") == "invoice_2026.pdf"

    def test_keeps_word_filename(self) -> None:
        assert sanitise_filename("Notice Draft.docx") == "Notice_Draft.docx"

    def test_empty_or_dot_falls_back_to_default(self) -> None:
        assert sanitise_filename("") == "upload.pdf"
        assert sanitise_filename(".") == "upload.pdf"
        assert sanitise_filename("..") == "upload.pdf"

    def test_unicode_and_control_chars_replaced(self) -> None:
        out = sanitise_filename("வரவு\x00report.pdf")
        # Control bytes never reach the saved name.
        assert "\x00" not in out
        assert out.endswith(".pdf")

    def test_extension_always_pdf(self) -> None:
        # Even when the caller lies about the extension we still write
        # it as a .pdf — the magic-bytes check inside `upload_pdf` is
        # the real gate; this just keeps the on-disk name predictable.
        assert sanitise_filename("doc.exe").endswith(".pdf")

    def test_long_filename_is_capped(self) -> None:
        long_stem = "a" * 500
        out = sanitise_filename(f"{long_stem}.pdf")
        assert len(out) <= 124  # 120 stem + ".pdf"


class TestValidateLanguage:
    @pytest.mark.parametrize("lang", ["auto", "en", "ta", "hi"])
    def test_accepted_languages_are_kept(self, lang: str) -> None:
        assert validate_language(lang) == lang

    @pytest.mark.parametrize("lang", ["", "xx", "../../en", "ta;rm -rf"])
    def test_unknown_languages_fall_back_to_auto(self, lang: str) -> None:
        assert validate_language(lang) == "auto"
