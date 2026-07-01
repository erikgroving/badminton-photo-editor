"""
Automatically detects the watermark region in edited photos by averaging the
absolute pixel difference between edited images and their corresponding raw
thumbnails across many pairs. The watermark appears as a consistent bright
region at the same location in every edited image.

Usage:
    python -m data.detect_watermark             # detect and print region
    python -m data.detect_watermark --apply     # also write region to config.py
    python -m data.detect_watermark --samples 100

Output: prints the detected WATERMARK_REGION tuple and saves a visual
        heatmap to sanity_check/watermark_heatmap.jpg.
"""
import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, MAPPING_FILE, THUMB_SIZE
from data.mapping import flat_entries, load_mapping
from data.raw_reader import extract_thumbnail


def detect(n_samples: int = 100) -> tuple[float, float, float, float] | None:
    """
    Returns (left, top, right, bottom) as fractions of image size, or None
    if the watermark couldn't be confidently located.
    """
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Run data/mapping.py first — {MAPPING_FILE} not found.")

    mapping = load_mapping()
    pairs   = [e for e in flat_entries(mapping) if e["label"] == 1 and e["edited"] and e["raw"]]

    rng    = random.Random(42)
    sample = rng.sample(pairs, min(n_samples, len(pairs)))
    print(f"Analysing {len(sample)} paired images for watermark location…")

    W, H   = THUMB_SIZE
    accum  = np.zeros((H, W), dtype=np.float64)
    count  = 0

    for entry in sample:
        try:
            raw_thumb = np.array(extract_thumbnail(entry["raw"], size=THUMB_SIZE)).astype(np.float32)

            edited = Image.open(entry["edited"]).convert("RGB")
            edited.thumbnail(THUMB_SIZE)
            canvas = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
            canvas.paste(edited, ((W - edited.width) // 2, (H - edited.height) // 2))
            ed_arr = np.array(canvas).astype(np.float32)

            diff   = np.mean(np.abs(ed_arr - raw_thumb), axis=2)
            accum += diff
            count += 1
        except Exception:
            continue

    if count == 0:
        print("No valid pairs processed.")
        return None

    avg_diff = accum / count

    # Save heatmap for visual inspection
    out_dir = BASE_DIR / "sanity_check"
    out_dir.mkdir(exist_ok=True)
    heatmap = (avg_diff / avg_diff.max() * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(out_dir / "watermark_heatmap.jpg")
    print(f"Heatmap saved: {out_dir / 'watermark_heatmap.jpg'}")

    # Find bounding box of the top-N% highest-difference pixels
    threshold = np.percentile(avg_diff, 95)
    mask      = avg_diff >= threshold
    rows      = np.any(mask, axis=1)
    cols      = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        print("Could not find a concentrated watermark region.")
        return None

    top,    bottom = np.argmax(rows), H - np.argmax(rows[::-1]) - 1
    left,   right  = np.argmax(cols), W - np.argmax(cols[::-1]) - 1

    # Add a small margin
    margin = 0.02
    l = max(0.0, left   / W - margin)
    t = max(0.0, top    / H - margin)
    r = min(1.0, right  / W + margin)
    b = min(1.0, bottom / H + margin)

    region = (round(l, 3), round(t, 3), round(r, 3), round(b, 3))
    print(f"\nDetected WATERMARK_REGION = {region}")
    print(f"  ({region[0]*100:.0f}% → {region[2]*100:.0f}% wide, "
          f"{region[1]*100:.0f}% → {region[3]*100:.0f}% tall)")
    print(f"\nInspect sanity_check/watermark_heatmap.jpg to verify this is correct.")
    return region


def apply_to_config(region: tuple) -> None:
    config_path = BASE_DIR / "config.py"
    text        = config_path.read_text(encoding="utf-8")
    new_line    = f"WATERMARK_REGION: tuple[float, float, float, float] | None = {region}"
    updated     = re.sub(
        r"WATERMARK_REGION:.*?= .*",
        new_line,
        text,
    )
    config_path.write_text(updated, encoding="utf-8")
    print(f"config.py updated: WATERMARK_REGION = {region}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--apply",   action="store_true",
                        help="Write detected region to config.py automatically")
    args = parser.parse_args()

    region = detect(args.samples)
    if region and args.apply:
        apply_to_config(region)
