"""Core OCR / search building blocks.

Submodules are *not* eagerly imported — that would force every consumer
(including unit tests of pure-functional modules like
:mod:`core.text_normalizer`) to install heavy native dependencies such
as ``PyMuPDF`` or ``paddleocr``. Import what you need explicitly.
"""
from __future__ import annotations

__all__: list[str] = []
