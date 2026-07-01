"""
Computes and caches interpretable per-image attributes used by the
attribute-aware culling model.

Each image → 8 float features (all normalised to roughly [0, 1]):

  0  global_blur        Laplacian variance of full thumbnail (higher = sharper)
  1  subject_sharpness  Tenengrad score on centre 50% crop (higher = sharper)
  2  exposure_mean      Mean luminance (0=black, 1=white; ~0.4-0.6 is good)
  3  exposure_contrast  Luminance std-dev (higher = more dynamic range)
  4  face_detected      1 if at least one face found, else 0
  5  face_sharpness     Laplacian variance inside the largest face bbox
                        (falls back to centre-crop Laplacian if no face)
  6  edge_crowding      Mean of the 4 border strips — how close the subject
                        is to the frame edge (higher = more edge crowding)
  7  highlight_clip     Fraction of pixels >245 in any channel (blown highlights)

Cache: data/attribute_cache.pkl  (keyed by (path, mtime))
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ATTR_DIM = 8
_CACHE_FILE = Path(__file__).parent / "attribute_cache.pkl"

# OpenCV Haar cascade for face detection (ships with opencv-python)
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(xml)
    return _FACE_CASCADE


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def extract_attributes(img_bgr: np.ndarray) -> np.ndarray:
    """
    img_bgr: uint8 BGR image (any size — will be resized to 224×224 internally).
    Returns float32 array of shape (ATTR_DIM,).
    """
    img = cv2.resize(img_bgr, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- 0: global blur ---
    raw_blur = _laplacian_var(gray)
    global_blur = np.tanh(raw_blur / 500.0)          # saturates ~1 around var=1000

    # --- 1: subject sharpness (centre 50% crop) ---
    cy, cx = h // 2, w // 2
    qh, qw = h // 4, w // 4
    centre_gray = gray[cy - qh: cy + qh, cx - qw: cx + qw]
    subject_sharpness = np.tanh(_tenengrad(centre_gray) / 1e6)

    # --- 2 & 3: exposure ---
    lum = gray.astype(np.float32) / 255.0
    exposure_mean     = float(lum.mean())
    exposure_contrast = float(lum.std())

    # --- 4 & 5: face detection ---
    faces = _get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    face_detected = 1.0 if len(faces) > 0 else 0.0
    if len(faces) > 0:
        # largest face by area
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        face_region = gray[fy: fy + fh, fx: fx + fw]
        face_sharpness = np.tanh(_laplacian_var(face_region) / 500.0)
    else:
        face_sharpness = global_blur     # fall back to whole-image sharpness

    # --- 6: edge crowding (how close bright subject is to the frame boundary) ---
    strip = max(1, h // 10)
    border_mean = np.mean([
        lum[:strip, :].mean(),
        lum[-strip:, :].mean(),
        lum[:, :strip].mean(),
        lum[:, -strip:].mean(),
    ])
    edge_crowding = float(border_mean)

    # --- 7: highlight clipping ---
    hi_frac = float(np.mean(img_bgr > 245))

    attrs = np.array([
        global_blur,
        subject_sharpness,
        exposure_mean,
        exposure_contrast,
        face_detected,
        face_sharpness,
        edge_crowding,
        hi_frac,
    ], dtype=np.float32)
    return attrs


# ── Cached batch extraction ───────────────────────────────────────────────────

def load_cache() -> dict:
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def get_attributes(raw_path: str) -> np.ndarray:
    """
    Returns cached attributes for a raw file (uses embedded thumbnail).
    Falls back to computing on-the-fly if not cached.
    """
    from data.raw_reader import extract_thumbnail
    from PIL import Image

    cache = load_cache()
    mtime = Path(raw_path).stat().st_mtime
    key   = (raw_path, mtime)
    if key in cache:
        return cache[key]

    pil_img = extract_thumbnail(raw_path, size=(224, 224))
    bgr     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    attrs   = extract_attributes(bgr)

    cache[key] = attrs
    save_cache(cache)
    return attrs


def build_attribute_cache(raw_paths: list[str], max_workers: int = 8) -> dict[str, np.ndarray]:
    """
    Pre-compute attributes for all raw_paths in parallel and return a
    {raw_path: attrs} dict.  Results are also persisted to the pickle cache.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    cache = load_cache()
    results: dict[str, np.ndarray] = {}
    to_compute: list[str] = []

    for p in raw_paths:
        mtime = Path(p).stat().st_mtime
        key   = (p, mtime)
        if key in cache:
            results[p] = cache[key]
        else:
            to_compute.append(p)

    print(f"Attribute cache: {len(results):,} hits, {len(to_compute):,} to compute")

    if to_compute:
        def _worker(path):
            from data.raw_reader import extract_thumbnail
            import cv2, numpy as np
            pil  = extract_thumbnail(path, size=(224, 224))
            bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return path, extract_attributes(bgr)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_worker, p): p for p in to_compute}
            for fut in tqdm(as_completed(futures), total=len(to_compute),
                            desc="Computing attributes"):
                path, attrs = fut.result()
                mtime = Path(path).stat().st_mtime
                cache[(path, mtime)] = attrs
                results[path] = attrs

        save_cache(cache)

    return results


ATTR_NAMES = [
    "global_blur",
    "subject_sharpness",
    "exposure_mean",
    "exposure_contrast",
    "face_detected",
    "face_sharpness",
    "edge_crowding",
    "highlight_clip",
]
