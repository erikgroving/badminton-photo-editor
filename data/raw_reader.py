"""
Utilities for reading Canon CR3 (and other rawpy-supported) raw files.

extract_thumbnail() — fast embedded JPEG preview; used for culling.
develop_raw()       — full neutral demosaic; used for crop and color models.

Both cache results to CACHE_DIR to avoid re-processing the same file twice.
Cache keys are based on the file's stem + mtime, so stale entries are evicted
when a file changes.
"""
import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, DEVELOP_SIZE, THUMB_SIZE, WATERMARK_REGION

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(path: Path, suffix: str) -> Path:
    stat = os.stat(path)
    digest = hashlib.md5(f"{path}{stat.st_mtime}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{path.stem}_{digest}_{suffix}.jpg"


def extract_thumbnail(
    raw_path: str | Path,
    size: tuple[int, int] = THUMB_SIZE,
    oriented: bool = False,
) -> Image.Image:
    """
    Extract the embedded JPEG thumbnail from a raw file and resize to `size`.
    Falls back to develop_raw() if no embedded thumb is available.
    Fast — does not demosaic the full sensor data. No disk caching.

    oriented=True: applies camera-orientation correction (_apply_flip).
    oriented=False: returns sensor-native orientation (used for model inference).
    """
    raw_path = Path(raw_path)

    try:
        import rawpy
        with rawpy.imread(str(raw_path)) as raw:
            flip = int(raw.sizes.flip)
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        else:
            img = Image.fromarray(thumb.data).convert("RGB")
    except Exception:
        img = develop_raw(raw_path, size=size)
        if oriented:
            img = _apply_flip(img, get_raw_flip(raw_path))
        return img

    if oriented:
        img = _apply_flip(img, flip)
        img.thumbnail((size[0], size[1]), Image.LANCZOS)
    else:
        img = _resize_cover(img, size)
    return img


def get_raw_flip(raw_path: str | Path) -> int:
    """
    Return the LibRaw orientation flip value for a raw file.
    Values: 0=normal, 3=180°, 5=90° CW, 6=90° CCW.
    Use _apply_flip() to rotate a PIL image accordingly.
    """
    raw_path = Path(raw_path)
    try:
        import rawpy
        with rawpy.imread(str(raw_path)) as raw:
            return int(raw.sizes.flip)
    except Exception:
        return 0


def _apply_flip(img: Image.Image, flip: int) -> Image.Image:
    """Rotate a PIL image (from extract_thumb raw JPEG bytes) to correct orientation.
    extract_thumb() returns the embedded JPEG in sensor-native landscape orientation
    WITHOUT applying any rotation — rawpy only applies flip in postprocess().
    For flip=5 (EXIF 8): head is on LEFT side of landscape → rotate 90° CCW to upright.
    For flip=6 (EXIF 6): head is on RIGHT side of landscape → rotate 90° CW to upright.
    NOTE: do NOT use this on postprocess() output — rawpy already corrects orientation there.
    """
    if flip == 3:  return img.rotate(180, expand=True)
    if flip == 5:  return img.rotate(90,  expand=True)  # EXIF 8: 0th row = visual right → rotate 90° CCW
    if flip == 6:  return img.rotate(270, expand=True)  # EXIF 6: 0th row = visual left  → rotate 90° CW
    return img


def develop_raw(
    raw_path: str | Path,
    size: tuple[int, int] = DEVELOP_SIZE,
    neutral: bool = True,
) -> Image.Image:
    """
    Demosaic the raw file and resize to fit within `size` (longest-edge bound,
    aspect ratio preserved). Does NOT apply camera orientation — rawpy always
    outputs sensor-native landscape. Use get_raw_flip() + _apply_flip() if you
    need the viewer-correct orientation.

    neutral=False uses camera white-balance + auto-brightness (matches camera JPEG).
    """
    raw_path = Path(raw_path)
    tag = f"dev_neutral_{size[0]}x{size[1]}" if neutral else f"dev_auto_{size[0]}x{size[1]}"
    cache_file = _cache_key(raw_path, tag)

    if cache_file.exists():
        return Image.open(cache_file).convert("RGB")

    import rawpy

    params = rawpy.Params(
        use_camera_wb=not neutral,
        use_auto_wb=False,
        no_auto_bright=neutral,
        output_bps=8,
        half_size=False,
    )
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(params)

    img = Image.fromarray(rgb).convert("RGB")
    # Preserve sensor aspect ratio (long edge ≤ max(size)) — no center-crop.
    img.thumbnail((max(size), max(size)), Image.LANCZOS)
    img.save(cache_file, format="JPEG", quality=92)
    return img


def mask_watermark(img: Image.Image, region: tuple | None = None) -> Image.Image:
    """
    Zero out the watermark region in a PIL Image (in-place copy).
    `region` is (left, top, right, bottom) as fractions of image dimensions.
    Falls back to config.WATERMARK_REGION if region is None.
    Returns the image unchanged if no region is configured.
    """
    region = region or WATERMARK_REGION
    if region is None:
        return img
    w, h = img.size
    l = int(region[0] * w)
    t = int(region[1] * h)
    r = int(region[2] * w)
    b = int(region[3] * h)
    out = img.copy()
    # Fill with black (0,0,0) — consistent between raw and edited
    from PIL import ImageDraw
    ImageDraw.Draw(out).rectangle([l, t, r, b], fill=(0, 0, 0))
    return out


def extract_thumbnail_ar(raw_path: str | Path, max_size: int = 512) -> Image.Image:
    """
    Like extract_thumbnail but preserves aspect ratio (no center-crop).
    The longest edge becomes max_size; the short edge is proportionally smaller.
    Used by the crop model where squishing to square is preferable to cropping
    off the sides (which would invalidate GT box coordinates).
    No disk caching.
    """
    raw_path = Path(raw_path)
    try:
        import rawpy
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        else:
            img = Image.fromarray(thumb.data).convert("RGB")
    except Exception:
        img = develop_raw(raw_path)

    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def _resize_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize to exactly `size` by center-cropping to the correct aspect ratio first."""
    target_w, target_h = size
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    if src_ratio > tgt_ratio:
        # wider than target: crop sides
        new_w = int(src_h * tgt_ratio)
        offset = (src_w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, src_h))
    elif src_ratio < tgt_ratio:
        # taller than target: crop top/bottom
        new_h = int(src_w / tgt_ratio)
        offset = (src_h - new_h) // 2
        img = img.crop((0, offset, src_w, offset + new_h))

    return img.resize(size, Image.LANCZOS)
