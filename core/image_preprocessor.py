"""Image preprocessing pipeline for OCR.

Engine-agnostic, pure-functional. Designed to maximise OCR quality on
**Tamil + English bilingual scans** without destroying fragile glyph
strokes (a common failure mode of aggressive binarisation on Tamil).

Pipeline (each stage is independently switchable):

    raw image
        → grayscale
        → border-crop  (remove black scan edges)
        → deskew       (Hough / projection profile)
        → denoise      (median filter, keeps glyph integrity)
        → contrast     (auto-stretch + optional CLAHE + sharpen)
        → binarize     (Sauvola adaptive — DISABLED by default for Tamil)
        → RGB output

The module uses OpenCV when available (``opencv-python-headless``) and
falls back to PIL-only operations when OpenCV is missing, so the backend
never crashes on a barebones install.

References
----------
* Sauvola, J. & Pietikäinen, M. (2000) — "Adaptive document image
  binarization". Pattern Recognition 33(2): 225–236.
* Zuiderveld, K. (1994) — "Contrast limited adaptive histogram
  equalization", Graphics Gems IV. Tamil scans benefit from CLAHE
  more than from naive auto-contrast.
* PaddleOCR best-practice DPI guidance: 300 dpi for Latin, 350-400 dpi
  for Indic scripts.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


# ── Tunables (no magic numbers in functions) ────────────────────────────────

#: Largest edge (in pixels) we'll keep on the input image. Anything bigger
#: is downscaled to save OCR time. 4500 px ≈ A4 at 540 dpi which is overkill.
MAX_EDGE_PIXELS: int = 4500

#: Auto-contrast removes this percentage of histogram tails on each side.
AUTOCONTRAST_CUTOFF_PCT: int = 1

#: Sharpness multiplier (1.0 = no change). Slight sharpen helps OCR boxes.
SHARPNESS_FACTOR: float = 1.25

#: Contrast multiplier (1.0 = no change).
CONTRAST_FACTOR: float = 1.20

#: Maximum skew angle (degrees) we attempt to correct. Beyond this we
#: assume the image is intentionally rotated and leave it alone.
MAX_SKEW_DEGREES: float = 10.0

#: Skew angles below this threshold (degrees) are not worth correcting.
MIN_SKEW_DEGREES: float = 0.2

#: Median filter window size. 3x3 is conservative — preserves Tamil
#: pulli (puḷḷi) and small superscript marks.
MEDIAN_FILTER_SIZE: int = 3

#: Sauvola window size (must be odd). 25 works well for ~350 dpi text.
SAUVOLA_WINDOW: int = 25

#: Sauvola sensitivity (0.2-0.5 typical).
SAUVOLA_K: float = 0.34

#: CLAHE clip limit. 2.0 brightens faded scans without amplifying noise
#: into Tamil pulli marks. Higher values risk blooming.
CLAHE_CLIP_LIMIT: float = 2.0

#: CLAHE tile grid. 8x8 is the standard Zuiderveld recommendation.
CLAHE_TILE_GRID: int = 8

#: Minimum dark-pixel ratio for the deskew Hough estimator to trust the
#: result. Below this we assume the page is mostly white space and skip.
MIN_INK_RATIO: float = 0.005


# ── Public dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreprocessOptions:
    """Configuration knobs for :func:`preprocess_for_ocr`.

    Each flag corresponds to a stage in the pipeline. The defaults match
    :class:`config.Settings` and are tuned for Tamil + English documents.
    """

    enable_border_crop: bool = True
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_contrast: bool = True
    enable_binarize: bool = False  # Off for Tamil safety; enable per scan if needed.
    enable_clahe: bool = True

    @classmethod
    def from_settings(cls, settings_obj) -> "PreprocessOptions":
        """Build :class:`PreprocessOptions` from a :class:`config.Settings`."""
        return cls(
            enable_border_crop=getattr(settings_obj, "enable_border_crop", True),
            enable_deskew=getattr(settings_obj, "enable_deskew", True),
            enable_denoise=getattr(settings_obj, "enable_denoise", True),
            enable_contrast=getattr(settings_obj, "enable_preprocessing", True),
            enable_binarize=getattr(settings_obj, "enable_binarize", False),
            enable_clahe=getattr(settings_obj, "enable_clahe", True),
        )

    def for_language(self, lang: Optional[str]) -> "PreprocessOptions":
        """Return a copy tuned for the given OCR language.

        Tamil benefits from CLAHE (fade-resistant contrast) but suffers
        from aggressive binarisation (it erodes pulli / fine strokes).
        Pure-Latin documents don't need CLAHE and can safely binarise
        when the user opted in.
        """
        if not lang:
            return self
        normalised = lang.lower()
        if normalised.startswith("ta"):
            return PreprocessOptions(
                enable_border_crop=self.enable_border_crop,
                enable_deskew=self.enable_deskew,
                enable_denoise=self.enable_denoise,
                enable_contrast=self.enable_contrast,
                enable_binarize=False,
                enable_clahe=True,
            )
        return self


# ── OpenCV is optional ──────────────────────────────────────────────────────

def _try_import_cv2():
    """Lazy-import OpenCV. Returns ``None`` when unavailable."""
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None


# ── Public entry point ──────────────────────────────────────────────────────

def preprocess_for_ocr(
    image: Image.Image,
    options: Optional[PreprocessOptions] = None,
) -> Image.Image:
    """Apply the full preprocessing pipeline to ``image``.

    Returns a new PIL :class:`Image.Image` in RGB mode, ready to be
    passed to any OCR engine. The original image is never mutated.

    Failures in any individual stage are logged and the previous stage's
    output is returned, so the pipeline never raises.
    """
    if image is None:
        raise ValueError("preprocess_for_ocr: image is None")

    opts = options or PreprocessOptions()

    try:
        gray = _to_grayscale(image)
        gray = _resize_if_huge(gray)

        if opts.enable_border_crop:
            gray = _safe(gray, _crop_dark_borders, "border_crop")

        if opts.enable_deskew:
            gray = _safe(gray, _deskew, "deskew")

        if opts.enable_denoise:
            gray = _safe(gray, _denoise, "denoise")

        if opts.enable_contrast:
            gray = _safe(gray, _enhance_contrast, "contrast")

        if opts.enable_clahe:
            gray = _safe(gray, _apply_clahe, "clahe")

        if opts.enable_binarize:
            gray = _safe(gray, _binarize_sauvola, "binarize")

        return gray.convert("RGB")
    except Exception as exc:  # noqa: BLE001 — last-ditch safety net
        logger.warning("Preprocessing pipeline failed (%s); returning original", exc)
        return image.convert("RGB") if image.mode != "RGB" else image


# ── Stage helpers ───────────────────────────────────────────────────────────

def _safe(
    img: Image.Image,
    fn,
    stage_name: str,
) -> Image.Image:
    """Run ``fn(img)``; on failure, log and return ``img`` unchanged."""
    try:
        out = fn(img)
        return out if out is not None else img
    except Exception as exc:  # noqa: BLE001 — pipeline must never crash
        logger.warning("Preprocessing stage '%s' failed: %s", stage_name, exc)
        return img


def _to_grayscale(image: Image.Image) -> Image.Image:
    """Convert to single-channel L mode."""
    return image if image.mode == "L" else image.convert("L")


def _resize_if_huge(image: Image.Image) -> Image.Image:
    """Cap the largest edge at :data:`MAX_EDGE_PIXELS` to control OCR runtime."""
    w, h = image.size
    longest = max(w, h)
    if longest <= MAX_EDGE_PIXELS:
        return image
    scale = MAX_EDGE_PIXELS / float(longest)
    new_size = (int(w * scale), int(h * scale))
    logger.debug("Resizing %sx%s → %sx%s for OCR", w, h, new_size[0], new_size[1])
    return image.resize(new_size, Image.LANCZOS)


def _crop_dark_borders(image: Image.Image) -> Image.Image:
    """Crop solid dark scan-edges that confuse layout / table detectors.

    Conservative: only crops if a contiguous dark frame is detected on
    at least two opposite edges.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return image

    threshold = max(30, int(arr.mean() * 0.25))
    mask = arr < threshold

    rows = mask.mean(axis=1)
    cols = mask.mean(axis=0)
    border_density = 0.85

    top = _leading_edge(rows, border_density)
    bottom = _trailing_edge(rows, border_density)
    left = _leading_edge(cols, border_density)
    right = _trailing_edge(cols, border_density)

    new_top = top
    new_bottom = arr.shape[0] - bottom
    new_left = left
    new_right = arr.shape[1] - right

    cropped_at_two_sides = (
        (top > 0 or bottom > 0) and (left > 0 or right > 0)
    )
    has_useful_area = (
        new_right - new_left > arr.shape[1] * 0.5
        and new_bottom - new_top > arr.shape[0] * 0.5
    )
    if cropped_at_two_sides and has_useful_area:
        return image.crop((new_left, new_top, new_right, new_bottom))
    return image


def _leading_edge(density: np.ndarray, threshold: float) -> int:
    """Return the first index where ``density`` drops below ``threshold``."""
    for idx, value in enumerate(density):
        if value < threshold:
            return idx
    return 0


def _trailing_edge(density: np.ndarray, threshold: float) -> int:
    """Return how many pixels of trailing high-density to trim."""
    count = 0
    for value in reversed(density):
        if value < threshold:
            break
        count += 1
    return count


def _deskew(image: Image.Image) -> Image.Image:
    """Detect skew angle and rotate the image flat.

    Uses OpenCV's Hough-line estimator when available, falling back to a
    projection-profile estimator (pure NumPy). Skew within
    :data:`MIN_SKEW_DEGREES` is treated as zero.
    """
    angle = _estimate_skew_angle(image)
    if abs(angle) < MIN_SKEW_DEGREES:
        return image
    if abs(angle) > MAX_SKEW_DEGREES:
        logger.debug("Skew %.2f° exceeds safe range; not rotating", angle)
        return image

    logger.debug("Deskew: rotating by %.2f°", -angle)
    return image.rotate(-angle, resample=Image.BICUBIC, expand=False, fillcolor=255)


def _estimate_skew_angle(image: Image.Image) -> float:
    """Return the dominant text-line skew angle in degrees."""
    cv2 = _try_import_cv2()
    if cv2 is not None:
        return _estimate_skew_cv2(image, cv2)
    return _estimate_skew_projection(image)


def _estimate_skew_cv2(image: Image.Image, cv2) -> float:
    """OpenCV-based skew estimation using HoughLinesP."""
    arr = np.asarray(image)
    inverted = 255 - arr  # text → bright
    if inverted.mean() / 255.0 < MIN_INK_RATIO:
        return 0.0

    edges = cv2.Canny(inverted, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 720,
        threshold=120,
        minLineLength=int(arr.shape[1] * 0.25),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -MAX_SKEW_DEGREES <= deg <= MAX_SKEW_DEGREES:
            angles.append(deg)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _estimate_skew_projection(image: Image.Image) -> float:
    """Pure-NumPy fallback skew estimator using horizontal projection profile."""
    arr = np.asarray(image, dtype=np.float32)
    inverted = 255.0 - arr
    if inverted.mean() / 255.0 < MIN_INK_RATIO:
        return 0.0

    best_angle = 0.0
    best_score = -math.inf
    angles_to_try = np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + 0.5, 0.5)

    for angle in angles_to_try:
        rotated = _rotate_array(inverted, float(angle))
        projection = rotated.sum(axis=1)
        # Sharper text → larger second-derivative variance.
        score = float(np.var(np.diff(projection)))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def _rotate_array(arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2-D array via PIL — keeps the dependency surface flat."""
    img = Image.fromarray(arr.astype(np.uint8))
    rotated = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0, expand=False)
    return np.asarray(rotated, dtype=np.float32)


def _denoise(image: Image.Image) -> Image.Image:
    """Median-filter denoising. Conservative window to preserve Tamil glyphs."""
    return image.filter(ImageFilter.MedianFilter(size=MEDIAN_FILTER_SIZE))


def _enhance_contrast(image: Image.Image) -> Image.Image:
    """Auto-contrast + gentle sharpen + contrast bump."""
    out = ImageOps.autocontrast(image, cutoff=AUTOCONTRAST_CUTOFF_PCT)
    out = ImageEnhance.Contrast(out).enhance(CONTRAST_FACTOR)
    out = ImageEnhance.Sharpness(out).enhance(SHARPNESS_FACTOR)
    return out


def _apply_clahe(image: Image.Image) -> Image.Image:
    """Contrast Limited Adaptive Histogram Equalisation.

    OpenCV is the only practical implementation; if it is not installed
    the stage is a no-op (the previous contrast pass has already done a
    competent job).
    """
    cv2 = _try_import_cv2()
    if cv2 is None:
        return image
    arr = np.asarray(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID),
    )
    equalised = clahe.apply(arr)
    return Image.fromarray(equalised, mode="L")


def _binarize_sauvola(image: Image.Image) -> Image.Image:
    """Adaptive Sauvola binarisation (preserves locally-varying ink density).

    Falls back to a global threshold if scikit-image is not installed.
    """
    arr = np.asarray(image, dtype=np.float64) / 255.0
    try:
        from skimage.filters import threshold_sauvola  # type: ignore

        thresh = threshold_sauvola(arr, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K)
        binary = (arr > thresh).astype(np.uint8) * 255
        return Image.fromarray(binary, mode="L")
    except ImportError:
        logger.debug("scikit-image not available; using Otsu fallback")
        return _binarize_otsu(image)


def _binarize_otsu(image: Image.Image) -> Image.Image:
    """Pure-NumPy Otsu thresholding."""
    arr = np.asarray(image)
    histogram, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    if total == 0:
        return image

    sum_total = float((histogram * np.arange(256)).sum())
    sum_b = 0.0
    weight_b = 0.0
    max_var = -math.inf
    threshold = 127

    for i in range(256):
        weight_b += histogram[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * histogram[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between > max_var:
            max_var = between
            threshold = i

    binary = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


# ── Diagnostics helper (used by tests / debug pages) ────────────────────────

def measure_image(image: Image.Image) -> Tuple[int, int, float]:
    """Return ``(width, height, ink_ratio)`` for an image.

    ``ink_ratio`` is the fraction of pixels darker than the per-image
    midpoint — a useful proxy for "how much text is on the page".
    """
    gray = _to_grayscale(image)
    arr = np.asarray(gray)
    if arr.size == 0:
        return (gray.size[0], gray.size[1], 0.0)
    midpoint = float(arr.mean())
    ink_ratio = float((arr < midpoint).mean())
    return (gray.size[0], gray.size[1], ink_ratio)
