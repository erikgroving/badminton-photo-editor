"""
Pre-compute image-region statistical features for every raw file in crop_gt.json.

Features (24 total) computed on a 128x128 grayscale thumbnail:
  [0-8]   Region mean brightness   — 3x3 grid, row-major (top-left to bottom-right)
  [9-17]  Region edge density      — Sobel edge magnitude mean per 3x3 region
  [18-21] Background space         — fraction of empty space above/below/left/right of player
            (falls back to [0.25, 0.25, 0.25, 0.25] when no player bbox available)
  [22]    Background color variance — std dev of pixel intensities in background mask
  [23]    Vertical balance          — (top_half_brightness - bottom_half_brightness) / 255

Cache format: { raw_path: [f0, f1, ..., f23] }
              null if image could not be read.

Run once before training:
    python data/cache_image_stats.py

Subsequent runs skip already-cached paths (incremental).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar

_CACHE_FILE        = Path(__file__).parent / "image_stats.json"
_PLAYER_BBOX_CACHE = Path(__file__).parent / "primary_player_bboxes.json"
_THUMB_SIZE        = 128   # fast enough; enough resolution for 3x3 region stats


# ── Feature computation ────────────────────────────────────────────────────────

def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Approximate Sobel edge magnitude on a 2-D uint8 array (pure numpy)."""
    g = gray.astype(np.float32)
    # Horizontal gradient (3x3 Sobel kernel, replicate-pad)
    gx = (
        -g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2]
        + g[:-2, 2:]  + 2 * g[1:-1, 2:]  + g[2:, 2:]
    )
    # Vertical gradient
    gy = (
        -g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:]
        + g[2:, :-2]  + 2 * g[2:, 1:-1]  + g[2:, 2:]
    )
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Pad back to original size (reflect border effect loses 1px each side)
    return np.pad(mag, 1, mode="constant", constant_values=0.0)


def compute_image_stats(
    raw_path: str,
    player_bbox: list | None = None,
) -> list[float] | None:
    """
    Compute 24 statistical features for one raw file.

    player_bbox: normalized [x1,y1,x2,y2] of primary player or None.
    Returns list of 24 floats, or None on error.
    """
    try:
        img = extract_thumbnail_ar(raw_path, max_size=_THUMB_SIZE)
    except Exception:
        return None

    # Resize to exact 128x128 (squish — consistent with crop model)
    img = img.resize((_THUMB_SIZE, _THUMB_SIZE), Image.LANCZOS)
    gray = np.array(img.convert("L"), dtype=np.float32)   # (128, 128)

    H, W = gray.shape  # both 128

    # ── Features 0-8: mean brightness per 3x3 region ─────────────────────────
    region_bright = []
    for ri in range(3):
        for ci in range(3):
            r0, r1 = ri * H // 3, (ri + 1) * H // 3
            c0, c1 = ci * W // 3, (ci + 1) * W // 3
            region_bright.append(float(gray[r0:r1, c0:c1].mean()) / 255.0)

    # ── Features 9-17: mean Sobel edge magnitude per 3x3 region ──────────────
    edges = _sobel_magnitude(gray)  # (128, 128), values in [0, ~1440]
    edges_norm = edges / (255.0 * np.sqrt(8))  # max theoretical ≈ 1440; normalise to ~[0,1]
    region_edges = []
    for ri in range(3):
        for ci in range(3):
            r0, r1 = ri * H // 3, (ri + 1) * H // 3
            c0, c1 = ci * W // 3, (ci + 1) * W // 3
            region_edges.append(float(edges_norm[r0:r1, c0:c1].mean()))

    # ── Features 18-21: background space (fraction) above/below/left/right ───
    if player_bbox is not None and any(v > 0 for v in player_bbox):
        px1, py1, px2, py2 = player_bbox
        space_left   = float(px1)
        space_top    = float(py1)
        space_right  = float(max(0.0, 1.0 - px2))
        space_bottom = float(max(0.0, 1.0 - py2))
    else:
        # No player detected — use uniform fallback (neutral, not zero)
        space_left = space_top = space_right = space_bottom = 0.25

    bg_space = [space_top, space_bottom, space_left, space_right]

    # ── Feature 22: background color variance ────────────────────────────────
    # Build a rough background mask by excluding the player bounding box region.
    if player_bbox is not None and any(v > 0 for v in player_bbox):
        px1, py1, px2, py2 = player_bbox
        mask = np.ones((H, W), dtype=bool)
        r0 = int(max(0, py1 * H))
        r1 = int(min(H, py2 * H))
        c0 = int(max(0, px1 * W))
        c1 = int(min(W, px2 * W))
        mask[r0:r1, c0:c1] = False
        bg_pixels = gray[mask]
    else:
        bg_pixels = gray.ravel()

    bg_var = float(bg_pixels.std()) / 255.0 if len(bg_pixels) > 0 else 0.0

    # ── Feature 23: vertical balance (top-heavy vs bottom-heavy) ─────────────
    top_half    = float(gray[:H // 2, :].mean()) / 255.0
    bottom_half = float(gray[H // 2:, :].mean()) / 255.0
    vert_balance = top_half - bottom_half   # positive = top brighter, negative = bottom brighter

    stats = region_bright + region_edges + bg_space + [bg_var, vert_balance]
    assert len(stats) == 24, f"Expected 24 features, got {len(stats)}"
    return stats


# ── Main caching loop ─────────────────────────────────────────────────────────

def main():
    if not CROP_GT_FILE.exists():
        print(f"ERROR: GT file not found: {CROP_GT_FILE}")
        sys.exit(1)

    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)

    raw_paths = [r["raw"] for r in all_records]
    print(f"Total raw files in crop_gt.json: {len(raw_paths)}")

    # Load primary player bbox cache for background-space features
    player_bbox_cache: dict = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        n_covered = sum(1 for p in raw_paths if player_bbox_cache.get(p) is not None)
        print(f"Primary player bbox cache: {len(player_bbox_cache)} entries "
              f"({n_covered}/{len(raw_paths)} raws covered)")
    else:
        print(f"  Warning: no primary player bbox cache at {_PLAYER_BBOX_CACHE}. "
              f"Background-space features will use uniform fallback (0.25).")

    # Load existing cache (incremental)
    stats_cache: dict = {}
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE) as fh:
            stats_cache = json.load(fh)
        print(f"Loaded {len(stats_cache)} existing cache entries")

    missing = [p for p in raw_paths if p not in stats_cache]
    print(f"Need to process: {len(missing)}")

    if not missing:
        print("All entries cached. Nothing to do.")
        return

    errors = 0
    for i, raw_path in enumerate(missing):
        if not Path(raw_path).exists():
            stats_cache[raw_path] = None
            errors += 1
        else:
            player_bbox = player_bbox_cache.get(raw_path)
            stats = compute_image_stats(raw_path, player_bbox)
            stats_cache[raw_path] = stats
            if stats is None:
                errors += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(missing):
            with open(_CACHE_FILE, "w") as fh:
                json.dump(stats_cache, fh, indent=2)
            pct = (i + 1) / len(missing) * 100
            print(f"  [{i+1}/{len(missing)}] {pct:.0f}%  errors so far: {errors}")

    with open(_CACHE_FILE, "w") as fh:
        json.dump(stats_cache, fh, indent=2)

    success = sum(1 for v in stats_cache.values() if v is not None)
    print(f"\nDone. {success} success / {errors} errors.")
    print(f"  Cache -> {_CACHE_FILE}")


if __name__ == "__main__":
    main()
