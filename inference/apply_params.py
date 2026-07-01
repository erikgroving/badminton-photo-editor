"""
Apply the predicted XMP color-correction parameters to a PIL image.

The color param model predicts the 14-vector defined by data.xmp_reader.PARAM_NAMES
(Temperature, Tint, Exposure2012, Contrast2012, …). This module maps those back
to actual pixel adjustments using the same names.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_PARAM_NAMES, COLOR_PARAM_RANGES


def vec_to_params(vec: list[float]) -> dict:
    """Denormalise a [-1, 1] model output vector → dict of named LR parameters."""
    params = {}
    for i, name in enumerate(COLOR_PARAM_NAMES):
        lo, hi = COLOR_PARAM_RANGES[name]
        params[name] = (vec[i] + 1) / 2 * (hi - lo) + lo
    return params


def apply_param_vec(img: Image.Image, param_vec: list[float]) -> Image.Image:
    """Apply a normalised model output vector to a PIL image. Returns corrected RGB image."""
    params = vec_to_params(param_vec)
    return apply_params_dict(img, params)


def apply_params_dict(img: Image.Image, params: dict) -> Image.Image:
    """Apply a dict of XMP-named parameters (raw LR scale values) to a PIL image."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    # Exposure (stops: EV = 2^x)
    exposure = params.get("Exposure2012", 0.0)
    arr = arr * (2.0 ** exposure)

    # Contrast: S-curve around midpoint
    c = params.get("Contrast2012", 0.0) / 100.0
    arr = (arr - 0.5) * (1.0 + c) + 0.5

    # Highlights: compress/expand upper tones
    hl = params.get("Highlights2012", 0.0) / 100.0
    mask_hi = np.clip(arr * 2.0 - 1.0, 0.0, 1.0)
    arr = arr + hl * mask_hi * (1.0 - arr) - hl * mask_hi * arr

    # Shadows: lift/lower lower tones
    sh = params.get("Shadows2012", 0.0) / 100.0
    mask_lo = np.clip(1.0 - arr * 2.0, 0.0, 1.0)
    arr = arr + sh * mask_lo * arr - sh * mask_lo * (1.0 - arr)

    # Whites / Blacks: clip-point adjustment
    w = params.get("Whites2012", 0.0) / 100.0
    b = params.get("Blacks2012", 0.0) / 100.0
    arr = arr + w * (1.0 - arr) * (arr > 0.75).astype(np.float32)
    arr = arr + b * arr         * (arr < 0.25).astype(np.float32)

    # Temperature (Kelvin): warm/cool via R↑B↓ or R↓B↑
    temp = params.get("Temperature", 5500.0)
    t = (temp - 5500.0) / 6000.0 * 0.12   # ±0.12 at extremes
    arr[..., 0] = np.clip(arr[..., 0] + t,  0.0, 1.0)   # R
    arr[..., 2] = np.clip(arr[..., 2] - t,  0.0, 1.0)   # B

    # Tint: green/magenta cast
    tint = params.get("Tint", 0.0)
    p = tint / 150.0 * 0.05
    arr[..., 1] = np.clip(arr[..., 1] + p, 0.0, 1.0)    # G

    # Saturation + Vibrance (vibrance = gentler saturation on already-saturated colours)
    sat = params.get("Saturation", 0.0) / 100.0
    vib = params.get("Vibrance",   0.0) / 100.0 * 0.5
    s   = 1.0 + sat + vib
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., np.newaxis]
    arr  = gray + s * (arr - gray)

    # Clarity / Texture / Dehaze: all require frequency-domain ops — skip for now
    # (Jay's values are usually small; the exposure/WB/tone adjustments dominate)

    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))


def default_correction(img: Image.Image) -> Image.Image:
    return img.copy()
