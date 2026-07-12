"""
Distill a predicted 3x4 color affine into Lightroom slider values for XMP export.

The affine model produces the best JPEG output but Lightroom needs sliders.
This fits the 14 XMP params (through a torch port of the calibrated renderer
in inference/apply_params.py) so that rendering the sliders approximates the
affine-corrected image. The sliders are a lossy approximation — the JPEG
pipeline uses the affine directly; the sidecar gives Jay a close starting
point in Lightroom.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_PARAM_NAMES, COLOR_PARAM_RANGES
from inference.apply_params import _CAL
from models.color_correction.affine_model import apply_affine_t, srgb_to_lab_t

_IDX = {n: i for i, n in enumerate(COLOR_PARAM_NAMES)}
_LO = torch.tensor([COLOR_PARAM_RANGES[n][0] for n in COLOR_PARAM_NAMES])
_HI = torch.tensor([COLOR_PARAM_RANGES[n][1] for n in COLOR_PARAM_NAMES])


def _render_sliders_t(img: torch.Tensor, vec: torch.Tensor,
                      camlog: float | None) -> torch.Tensor:
    """Torch port of apply_params_dict. img: [3,H,W] in [0,1]; vec: normalized
    [-1,1] 14-vector. Mirrors the calibrated numpy renderer exactly."""
    lo, hi = _LO.to(vec.device), _HI.to(vec.device)
    p = (vec.clamp(-1, 1) + 1) / 2 * (hi - lo) + lo   # LR-scale params

    arr = img.permute(1, 2, 0)                        # [H,W,3]

    exposure = p[_IDX["Exposure2012"]] * _CAL["k_exp"]
    lin = arr.clamp(1e-6, 1.0) ** 2.2
    arr = (lin * (2.0 ** exposure)).clamp(1e-6, 4.0) ** (1 / 2.2)

    c = p[_IDX["Contrast2012"]] / 100.0
    arr = (arr - 0.5) * (1.0 + c) + 0.5

    hl = p[_IDX["Highlights2012"]] / 100.0
    mask_hi = (arr * 2.0 - 1.0).clamp(0.0, 1.0)
    arr = arr + hl * mask_hi * (1.0 - arr) - hl * mask_hi * arr

    sh = p[_IDX["Shadows2012"]] / 100.0
    mask_lo = (1.0 - arr * 2.0).clamp(0.0, 1.0)
    arr = arr + sh * mask_lo * arr - sh * mask_lo * (1.0 - arr)

    w = p[_IDX["Whites2012"]] / 100.0
    b = p[_IDX["Blacks2012"]] / 100.0
    arr = arr + w * (1.0 - arr) * (arr > 0.75).float()
    arr = arr + b * arr * (arr < 0.25).float()

    if camlog is not None:
        temp = p[_IDX["Temperature"]].clamp(min=1500.0)
        mired_shot = _CAL["A"] + _CAL["B"] * camlog
        t = _CAL["k_temp"] * (1e6 / temp - mired_shot)
        arr = torch.stack([arr[..., 0] + t, arr[..., 1], arr[..., 2] - t], dim=-1)

    tint = p[_IDX["Tint"]]
    g = _CAL["k_tint"] * (tint - _CAL["tint0"]) / 150.0
    arr = torch.stack([arr[..., 0], arr[..., 1] + g, arr[..., 2]], dim=-1)

    sat = p[_IDX["Saturation"]] / 100.0
    vib = p[_IDX["Vibrance"]] / 100.0 * 0.5
    s = 1.0 + sat + vib
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]).unsqueeze(-1)
    arr = gray + s * (arr - gray)

    return arr.clamp(0.0, 1.0).permute(2, 0, 1)


def distill_sliders(image: Image.Image, M: np.ndarray,
                    camlog: float | None, device,
                    size: int = 128, steps: int = 250,
                    lr: float = 0.05) -> tuple[dict, float]:
    """Fit sliders so slider-render(image) ~ affine-render(image).

    Returns (params dict on LR scale, residual Lab-L1 between the two renders).
    """
    small = image.convert("RGB").resize((size, size), Image.LANCZOS)
    img = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0
                           ).permute(2, 0, 1).to(device)
    with torch.no_grad():
        target = apply_affine_t(img.unsqueeze(0),
                                torch.from_numpy(M.astype(np.float32))
                                .unsqueeze(0).to(device))[0]
        lab_target = srgb_to_lab_t(target.permute(1, 2, 0))

    vec = torch.zeros(len(COLOR_PARAM_NAMES), device=device, requires_grad=True)
    opt = torch.optim.Adam([vec], lr=lr)
    for _ in range(steps):
        out = _render_sliders_t(img, vec, camlog)
        loss = (srgb_to_lab_t(out.permute(1, 2, 0)) - lab_target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vec.clamp_(-1.0, 1.0)

    with torch.no_grad():
        residual = float((srgb_to_lab_t(_render_sliders_t(img, vec, camlog)
                                        .permute(1, 2, 0)) - lab_target)
                         .abs().mean().item())
        lo, hi = _LO.to(device), _HI.to(device)
        p = ((vec.clamp(-1, 1) + 1) / 2 * (hi - lo) + lo).cpu().tolist()
    return dict(zip(COLOR_PARAM_NAMES, p)), residual
