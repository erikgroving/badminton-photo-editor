"""
Extracts ground-truth color correction parameters from paired raw/edited images.

For each pair (after crop alignment), uses scipy Nelder-Mead to find the 9
Lightroom-style parameters that minimise the MSE between:
  - the adjusted neutral-developed raw crop
  - the edited JPEG crop

Results cached to COLOR_GT_FILE so this expensive process runs only once.

Usage:
    python -m data.color_analyzer [--workers 4]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_GT_FILE, COLOR_PARAM_NAMES, COLOR_PARAM_RANGES, COLOR_SIZE, CROP_GT_FILE
from data.raw_reader import develop_raw


# ── Parameter application (pure numpy) ───────────────────────────────────────

def _apply_params(img_np: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply Lightroom-style adjustments to a float32 RGB image in [0,1].
    Returns float32 in [0,1].
    """
    img = img_np.astype(np.float32)

    # Exposure: multiply by 2^stops
    img = img * (2.0 ** params["exposure"])

    # Contrast: S-curve around midpoint 0.5
    c = params["contrast"] / 100.0
    img = (img - 0.5) * (1.0 + c) + 0.5

    # Highlights/Shadows: apply to upper/lower tonal range
    hl = params["highlights"] / 100.0
    sh = params["shadows"]    / 100.0
    mask_hi = np.clip(img * 2.0 - 1.0, 0.0, 1.0)
    mask_lo = np.clip(1.0 - img * 2.0, 0.0, 1.0)
    img = img + hl * mask_hi * (1.0 - img) - hl * mask_hi * img
    img = img + sh * mask_lo * img         - sh * mask_lo * (1.0 - img)

    # Whites / Blacks
    w = params["whites"] / 100.0
    b = params["blacks"] / 100.0
    img = img + w * (1.0 - img) * (img > 0.75).astype(np.float32)
    img = img + b * img         * (img < 0.25).astype(np.float32)

    # Temperature / Tint (shift in YCbCr-like space)
    # temp_shift > 0 → warmer (more red/less blue), < 0 → cooler
    t = params["temp_shift"] / 50.0  # normalise to [-1, 1]
    p = params["tint_shift"] / 50.0
    img[..., 0] = np.clip(img[..., 0] + t * 0.1,  0, 1)   # R
    img[..., 1] = np.clip(img[..., 1] + p * 0.05, 0, 1)   # G
    img[..., 2] = np.clip(img[..., 2] - t * 0.1,  0, 1)   # B

    # Saturation
    s = 1.0 + params["saturation"] / 100.0
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gray = gray[..., np.newaxis]
    img  = gray + s * (img - gray)

    return np.clip(img, 0.0, 1.0)


def params_to_vec(params: dict) -> list[float]:
    vec = []
    for name in COLOR_PARAM_NAMES:
        lo, hi = COLOR_PARAM_RANGES[name]
        vec.append((params[name] - lo) / (hi - lo) * 2 - 1)  # → [-1, 1]
    return vec


def vec_to_params(vec: list[float]) -> dict:
    params = {}
    for i, name in enumerate(COLOR_PARAM_NAMES):
        lo, hi = COLOR_PARAM_RANGES[name]
        params[name] = (vec[i] + 1) / 2 * (hi - lo) + lo
    return params


def default_params() -> dict:
    return {name: 0.0 for name in COLOR_PARAM_NAMES}


# ── Optimizer ─────────────────────────────────────────────────────────────────

def _fit_params(raw_crop_np: np.ndarray, edited_crop_np: np.ndarray) -> dict:
    """
    Find the 9 parameters that minimise MSE(apply(raw, p), edited).
    Both inputs are float32 in [0,1], shape (H, W, 3).
    """
    from scipy.optimize import minimize

    x0 = np.zeros(len(COLOR_PARAM_NAMES))

    def objective(x):
        params = vec_to_params(x.tolist())
        adjusted = _apply_params(raw_crop_np, params)
        return float(np.mean((adjusted - edited_crop_np) ** 2))

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
    return vec_to_params(np.clip(result.x, -1.0, 1.0).tolist())


def _process_record(record: dict) -> dict | None:
    try:
        from PIL import Image

        # box = [x1, y1, x2, y2] normalized to [0, 1] in raw thumbnail space
        x1, y1, x2, y2 = record["box"]
        angle_deg = record.get("angle_deg", 0.0)

        raw_pil    = develop_raw(record["raw"], size=COLOR_SIZE, neutral=False)
        edited_pil = Image.open(record["edited"]).convert("RGB")

        # Crop the developed raw at the GT box
        W, H   = raw_pil.size   # == COLOR_SIZE
        left   = int(x1 * W)
        upper  = int(y1 * H)
        right  = int(x2 * W)
        lower  = int(y2 * H)
        right  = max(right,  left + 1)
        lower  = max(lower,  upper + 1)
        raw_crop = raw_pil.crop((left, upper, right, lower))

        # Rotate crop to match edited orientation (angle_deg is CCW, per PIL convention)
        if abs(angle_deg) > 45.0:
            raw_crop = raw_crop.rotate(round(angle_deg), expand=True, resample=Image.BILINEAR)

        # Resize both to COLOR_SIZE for parameter fitting
        raw_crop   = raw_crop.resize(COLOR_SIZE, Image.LANCZOS)
        edited_pil = edited_pil.resize(COLOR_SIZE, Image.LANCZOS)

        raw_np    = np.array(raw_crop).astype(np.float32) / 255.0
        edited_np = np.array(edited_pil).astype(np.float32) / 255.0

        fitted = _fit_params(raw_np, edited_np)
        mse    = float(np.mean((_apply_params(raw_np, fitted) - edited_np) ** 2))

        return {
            "raw":    record["raw"],
            "edited": record["edited"],
            "params": fitted,
            "mse":    mse,
        }
    except Exception:
        return None


def build_color_gt(workers: int = 4) -> list[dict]:
    if not CROP_GT_FILE.exists():
        raise FileNotFoundError(f"Run data/crop_detector.py first — {CROP_GT_FILE} not found.")

    with open(CROP_GT_FILE) as fh:
        records = json.load(fh)

    print(f"Fitting color params for {len(records):,} pairs…  (this is slow — run once and cache)")
    results = []
    failed  = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_record, r): r for r in records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Color fitting"):
            r = fut.result()
            if r:
                results.append(r)
            else:
                failed += 1

    print(f"Success: {len(results):,}  |  Failed: {failed:,}")
    avg_mse = sum(r["mse"] for r in results) / len(results) if results else 0
    print(f"Average MSE: {avg_mse:.6f}")

    COLOR_GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COLOR_GT_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved color ground truth: {COLOR_GT_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build_color_gt(workers=args.workers)
