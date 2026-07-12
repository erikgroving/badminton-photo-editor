"""
Apply the predicted XMP color-correction parameters to a PIL image.

The color param model predicts the 14-vector defined by data.xmp_reader.PARAM_NAMES
(Temperature, Tint, Exposure2012, Contrast2012, …). This module maps those back
to actual pixel adjustments using the same names.

WHITE BALANCE SEMANTICS (calibrated 2026-07-12): XMP Temperature is an ABSOLUTE
Kelvin value, but the image being adjusted is a develop_raw(neutral=False) render
that is already camera-white-balanced. Temperature is therefore applied as a
DIFFERENTIAL in mired space against the raw's as-shot white balance (pass
`camlog` — see data.raw_reader.get_as_shot_camlog). Without as-shot info the
temperature shift is skipped, which measures within noise of the calibrated
differential (val ΔE2000 11.89 vs 12.02) and far better than the old
shift-from-5500K interpretation (14.72), which double-corrected WB.

Renderer constants below were fitted by minimizing GT-slider-replay ΔE2000
against the photographer's actual edits on 100 train pairs
(scratchpad/calibrate_color_renderer.py), validated on 100 val pairs.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_PARAM_NAMES, COLOR_PARAM_RANGES

# Calibrated renderer constants (fit 2026-07-12, train ΔE 15.78 → 12.90)
_CAL = {
    "k_exp":  1.26836,    # exposure scale, applied in linear space
    "k_temp": 0.0017,     # R/B shift per mired of (predicted − as-shot)
    "A":      253.98472,  # as-shot mired = A + B·camlog
    "B":      -56.94468,
    "k_tint": 0.05343,    # G shift per 150 tint units, offset by tint0
    "tint0":  18.43425,
}


def vec_to_params(vec: list[float]) -> dict:
    """Denormalise a [-1, 1] model output vector → dict of named LR parameters."""
    params = {}
    for i, name in enumerate(COLOR_PARAM_NAMES):
        lo, hi = COLOR_PARAM_RANGES[name]
        params[name] = (vec[i] + 1) / 2 * (hi - lo) + lo
    return params


def apply_param_vec(img: Image.Image, param_vec: list[float],
                    camlog: float | None = None) -> Image.Image:
    """Apply a normalised model output vector to a PIL image. Returns corrected RGB image."""
    params = vec_to_params(param_vec)
    return apply_params_dict(img, params, camlog=camlog)


def apply_params_dict(img: Image.Image, params: dict,
                      camlog: float | None = None) -> Image.Image:
    """Apply a dict of XMP-named parameters (raw LR scale values) to a PIL image.

    camlog: ln(as-shot R gain / B gain) from the raw's camera white balance —
    enables the differential Temperature application. None → skip temp shift.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    # Exposure (stops) — in LINEAR space, calibrated scale
    exposure = params.get("Exposure2012", 0.0) * _CAL["k_exp"]
    if exposure != 0.0:
        lin = np.clip(arr, 1e-6, 1.0) ** 2.2
        arr = np.clip(lin * (2.0 ** exposure), 0.0, 4.0) ** (1 / 2.2)

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

    # Temperature: differential in mired vs as-shot WB (see module docstring)
    if camlog is not None:
        temp = params.get("Temperature", 5500.0)
        mired_pred = 1e6 / max(temp, 1500.0)
        mired_shot = _CAL["A"] + _CAL["B"] * camlog
        t = _CAL["k_temp"] * (mired_pred - mired_shot)
        arr[..., 0] = arr[..., 0] + t   # R
        arr[..., 2] = arr[..., 2] - t   # B

    # Tint: green/magenta cast, calibrated offset
    tint = params.get("Tint", 0.0)
    p = _CAL["k_tint"] * (tint - _CAL["tint0"]) / 150.0
    arr[..., 1] = arr[..., 1] + p       # G

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
