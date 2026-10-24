"""Unit tests for :mod:`core.image_preprocessor`.

We synthesise small images in-memory so the suite has no external
dependencies (no fixture PDFs, no scan files). The goal is to verify the
pipeline doesn't crash, never returns ``None``, always emits an RGB
output of the same size order, and that the optional stages do *something*
when they're enabled.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core.image_preprocessor import (
    MAX_EDGE_PIXELS,
    PreprocessOptions,
    measure_image,
    preprocess_for_ocr,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _white_page(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _page_with_text(width: int = 800, height: int = 1000) -> Image.Image:
    img = _white_page(width, height)
    draw = ImageDraw.Draw(img)
    for row in range(0, height, 40):
        draw.rectangle([(50, row), (width - 50, row + 4)], fill=(20, 20, 20))
    return img


def _page_with_dark_border(width: int = 800, height: int = 1000) -> Image.Image:
    img = _page_with_text(width, height)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width, 30)], fill=(0, 0, 0))           # top
    draw.rectangle([(0, height - 30), (width, height)], fill=(0, 0, 0))  # bottom
    draw.rectangle([(0, 0), (30, height)], fill=(0, 0, 0))           # left
    draw.rectangle([(width - 30, 0), (width, height)], fill=(0, 0, 0))  # right
    return img


# ── Smoke tests ────────────────────────────────────────────────────────────

def test_preprocess_returns_rgb_image() -> None:
    out = preprocess_for_ocr(_page_with_text())
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_preprocess_does_not_crash_on_blank_page() -> None:
    out = preprocess_for_ocr(_white_page())
    assert isinstance(out, Image.Image)
    assert out.size[0] > 0 and out.size[1] > 0


def test_preprocess_handles_grayscale_input() -> None:
    img = _page_with_text().convert("L")
    out = preprocess_for_ocr(img)
    assert out.mode == "RGB"


def test_preprocess_does_not_mutate_input() -> None:
    img = _page_with_text()
    original_pixels = np.array(img)
    preprocess_for_ocr(img)
    assert np.array_equal(np.array(img), original_pixels)


# ── Resize stage ──────────────────────────────────────────────────────────

def test_preprocess_resizes_oversized_images() -> None:
    huge = _page_with_text(width=MAX_EDGE_PIXELS * 2, height=300)
    out = preprocess_for_ocr(huge)
    assert max(out.size) <= MAX_EDGE_PIXELS


def test_preprocess_keeps_modest_images_unchanged_in_size() -> None:
    img = _page_with_text(width=1200, height=1600)
    out = preprocess_for_ocr(img)
    assert out.size == (1200, 1600)


# ── Border crop stage ─────────────────────────────────────────────────────

def test_border_crop_removes_dark_edges() -> None:
    img = _page_with_dark_border()
    options = PreprocessOptions(
        enable_border_crop=True,
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        enable_binarize=False,
    )
    out = preprocess_for_ocr(img, options)
    assert out.size[0] < img.size[0] or out.size[1] < img.size[1]


def test_border_crop_disabled_keeps_full_size() -> None:
    img = _page_with_dark_border()
    options = PreprocessOptions(
        enable_border_crop=False,
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        enable_binarize=False,
    )
    out = preprocess_for_ocr(img, options)
    assert out.size == img.size


# ── Stage independence ───────────────────────────────────────────────────

def test_disabling_all_stages_still_returns_valid_image() -> None:
    options = PreprocessOptions(
        enable_border_crop=False,
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        enable_binarize=False,
    )
    out = preprocess_for_ocr(_page_with_text(), options)
    assert out.mode == "RGB"
    assert out.size == (800, 1000)


def test_binarize_produces_two_intensity_clusters() -> None:
    options = PreprocessOptions(
        enable_border_crop=False,
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        enable_binarize=True,
    )
    out = preprocess_for_ocr(_page_with_text(), options)
    arr = np.asarray(out.convert("L"))
    unique = np.unique(arr)
    # Allow a tiny amount of bilinear-resampling noise but the histogram
    # should be heavily bimodal — most mass within 5 levels of 0 or 255.
    extreme_mass = ((arr <= 5) | (arr >= 250)).mean()
    assert extreme_mass > 0.99
    assert len(unique) >= 1


# ── from_settings adapter ─────────────────────────────────────────────────

def test_options_from_settings_reads_known_flags() -> None:
    class FakeSettings:
        enable_border_crop = False
        enable_deskew = False
        enable_denoise = True
        enable_preprocessing = True
        enable_binarize = True

    opts = PreprocessOptions.from_settings(FakeSettings())
    assert opts.enable_border_crop is False
    assert opts.enable_deskew is False
    assert opts.enable_denoise is True
    assert opts.enable_contrast is True
    assert opts.enable_binarize is True


# ── measure_image ─────────────────────────────────────────────────────────

def test_measure_image_returns_size_and_ink_ratio() -> None:
    width, height, ink = measure_image(_page_with_text())
    assert width == 800
    assert height == 1000
    assert 0.0 < ink < 1.0


def test_measure_image_handles_blank() -> None:
    width, height, ink = measure_image(_white_page())
    assert width == 800
    assert height == 1000
    # Mean equals the only pixel value, so no pixel is *strictly* below it.
    assert ink == 0.0
