"""
Extracts ground-truth crop boxes from raw->edited pairs using SIFT homography.

For each pair:
  1. Extract raw thumbnail (768 px) and load edited JPG (resized to fit 768)
  2. Mask the watermark region in the edited image before SIFT
  3. Match SIFT keypoints, compute homography (edited -> raw thumbnail space)
  4. Transform the 4 corners of the edited back into raw thumbnail coordinates
  5. Record normalized [x1, y1, x2, y2] in [0,1] + rotation angle

Output saved to CROP_GT_FILE (data/crop_gt.json).

Usage:
    python -m data.crop_detector            # build (skip if file exists)
    python -m data.crop_detector --rebuild  # force rebuild
    python -m data.crop_detector --verify   # print statistics
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROP_GT_FILE, CROP_MIN_INLIER_RATIO, WATERMARK_REGION
from data.mapping import flat_entries, load_mapping
from data.raw_reader import extract_thumbnail_ar

SIFT_SIZE   = 768
MIN_MATCHES = 12
MIN_INLIERS = 8
RATIO_TEST  = 0.72
MAX_WORKERS = 6


def _thumb_gray(raw_path: str) -> tuple[np.ndarray, int, int]:
    img = extract_thumbnail_ar(raw_path, max_size=SIFT_SIZE)
    return np.array(img.convert("L")), img.width, img.height


def _edited_gray(edited_path: str, mask_watermark: bool = True) -> tuple[np.ndarray, float]:
    """Load edited JPG, optionally mask watermark, resize to fit SIFT_SIZE."""
    img = Image.open(edited_path).convert("RGB")
    orig_aspect = img.width / img.height

    if mask_watermark and WATERMARK_REGION is not None:
        x0, y0, x1, y1 = WATERMARK_REGION
        px0 = int(x0 * img.width)
        py0 = int(y0 * img.height)
        px1 = int(x1 * img.width)
        py1 = int(y1 * img.height)
        arr = np.array(img)
        arr[py0:py1, px0:px1] = 0
        img = Image.fromarray(arr)

    img.thumbnail((SIFT_SIZE, SIFT_SIZE), Image.LANCZOS)
    return np.array(img.convert("L")), orig_aspect


def detect_crop(raw_path: str, edited_path: str) -> dict | None:
    """
    Returns crop dict or None if matching fails.

    box: [x1, y1, x2, y2] normalized to [0, 1] in raw thumbnail space.
    angle_deg: rotation of the edited crop relative to axis-aligned.
    """
    try:
        raw_gray, raw_w, raw_h = _thumb_gray(raw_path)
        edit_gray, edit_aspect = _edited_gray(edited_path)
        edit_h, edit_w = edit_gray.shape

        sift = cv2.SIFT_create(nfeatures=2000)
        kp_e, des_e = sift.detectAndCompute(edit_gray, None)
        kp_r, des_r = sift.detectAndCompute(raw_gray, None)

        if (des_e is None or des_r is None
                or len(kp_e) < MIN_MATCHES or len(kp_r) < MIN_MATCHES):
            return None

        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(des_e, des_r, k=2)
        good = [m for m, n in raw_matches
                if len([m, n]) == 2 and m.distance < RATIO_TEST * n.distance]

        if len(good) < MIN_MATCHES:
            return None

        pts_e = np.float32([kp_e[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_r = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_e, pts_r, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None

        n_inliers    = int(mask.sum())
        inlier_ratio = n_inliers / len(mask)

        if n_inliers < MIN_INLIERS or inlier_ratio < CROP_MIN_INLIER_RATIO:
            return None

        # Project edited corners -> raw thumbnail space
        corners = np.float32([
            [0, 0], [edit_w, 0], [edit_w, edit_h], [0, edit_h]
        ]).reshape(-1, 1, 2)
        tr = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        x1 = float(np.clip(tr[:, 0].min() / raw_w, 0.0, 1.0))
        y1 = float(np.clip(tr[:, 1].min() / raw_h, 0.0, 1.0))
        x2 = float(np.clip(tr[:, 0].max() / raw_w, 0.0, 1.0))
        y2 = float(np.clip(tr[:, 1].max() / raw_h, 0.0, 1.0))

        if x2 - x1 < 0.05 or y2 - y1 < 0.05:
            return None

        top_edge  = tr[1] - tr[0]
        angle_deg = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])))

        return {
            "box":           [x1, y1, x2, y2],
            "angle_deg":     round(angle_deg, 3),
            "aspect":        round(edit_aspect, 4),
            "inlier_ratio":  round(inlier_ratio, 3),
            "n_matches":     n_inliers,
        }
    except Exception:
        return None


def build_crop_gt(mapping: dict) -> list[dict]:
    pairs = [e for e in flat_entries(mapping) if e["label"] == 1 and e["edited"] and e["raw"]]
    print(f"Processing {len(pairs):,} raw->edited pairs…")

    results: list[dict] = []
    failed = 0

    def _worker(e):
        r = detect_crop(e["raw"], e["edited"])
        if r is None:
            return None
        return {**r, "raw": e["raw"], "edited": e["edited"], "split": e["split"]}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_worker, e): e for e in pairs}
        for fut in tqdm(as_completed(futs), total=len(pairs), desc="SIFT homography"):
            r = fut.result()
            if r is None:
                failed += 1
            else:
                results.append(r)

    total = len(results) + failed
    print(f"\nSuccessful: {len(results):,}  |  Failed: {failed:,}  ({len(results)/total:.1%} yield)")
    for sp in ("train", "val", "test"):
        n = sum(1 for r in results if r["split"] == sp)
        print(f"  {sp:5s}: {n:,}")
    return results


def _verify(annotations: list[dict]) -> None:
    boxes   = np.array([a["box"] for a in annotations])
    widths  = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    aspects = np.array([a["aspect"] for a in annotations])
    angles  = np.array([a["angle_deg"] for a in annotations])

    print(f"\n{len(annotations):,} annotations")
    print(f"Crop width  (0-1):  mean={widths.mean():.3f}  std={widths.std():.3f}")
    print(f"Crop height (0-1):  mean={heights.mean():.3f}  std={heights.std():.3f}")
    print(f"Aspect ratio:       mean={aspects.mean():.3f}  std={aspects.std():.3f}")
    print(f"Rotation angle:     mean={angles.mean():.2f}°  std={angles.std():.2f}°")
    inliers = [a["inlier_ratio"] for a in annotations]
    print(f"Inlier ratio:       mean={np.mean(inliers):.3f}")

    bins   = [0.0, 0.7, 0.9, 1.1, 1.4, 1.8, 99]
    labels = ["portrait <0.7", "square-ish", "3:2 landscape", "4:3-ish", "16:9+", "ultra-wide"]
    print("\nAspect ratio distribution:")
    for i, label in enumerate(labels):
        n = ((aspects >= bins[i]) & (aspects < bins[i+1])).sum()
        print(f"  {label:18s}: {n:4d}  ({n/len(aspects):.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verify",  action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    if CROP_GT_FILE.exists() and not args.rebuild:
        print(f"Crop GT already exists: {CROP_GT_FILE}  (use --rebuild to regenerate)")
        with open(CROP_GT_FILE) as fh:
            annotations = json.load(fh)
    else:
        mapping = load_mapping()
        annotations = build_crop_gt(mapping)
        CROP_GT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CROP_GT_FILE, "w") as fh:
            json.dump(annotations, fh, indent=2)
        print(f"Saved: {CROP_GT_FILE}")

    if args.verify:
        _verify(annotations)
