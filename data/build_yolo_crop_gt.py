"""
Build YOLO-anchored crop ground truth for model retraining.

Replaces SIFT-homography GT (crop_gt.json) with player-detected portrait crops.
For each raw in the existing GT:
  1. Extract aspect-ratio-preserving thumbnail (same as model training input)
  2. Run YOLO11n person detection
  3. Compute a portrait 2:3 crop anchored to the primary player
  4. Normalize the box to [0, 1] in thumbnail space

Key differences from crop_gt.json:
  - angle_deg is always 0 — rawpy already applies camera orientation, so the
    thumbnail is viewer-correct; no rotation is needed in the crop target
  - Box is derived from player position, not Jay's Lightroom edit, so it
    guarantees full-body inclusion with consistent composition rules:
      - player fills 60% of crop height
      - player head at 22% from top (rule of thirds)
      - portrait 2:3 aspect ratio throughout

When YOLO cannot detect a player, the original GT box is kept as a fallback
(flagged with yolo_detected=False). Pass --fallback-skip to drop those raws.

Output: data/crop_gt_yolo.json  (same schema as crop_gt.json)

Usage:
    python -m data.build_yolo_crop_gt
    python -m data.build_yolo_crop_gt --fallback-skip
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHECKPOINTS_DIR, CROP_GT_FILE, CROP_GT_YOLO_FILE
from data.raw_reader import extract_thumbnail_ar

_EXTRACT_SIZE = 512   # must match CropDataset in models/cropping/train.py

# Mirror inference/run.py _compute_player_crop parameters exactly so training
# and inference targets are identical.
_PLAYER_FILL  = 0.60  # player body fills 60% of crop height
_HEAD_ROOM    = 0.22  # top-of-player-bbox at 22% from crop top (rule of thirds)
_MIN_FRACTION = 0.02  # ignore detections smaller than 2% of image area


def _player_crop_normalized(
    boxes: list[tuple],
    W: int,
    H: int,
) -> list[float] | None:
    """
    Compute a player-anchored portrait (2:3) crop and return it as
    normalized [x1, y1, x2, y2] in [0, 1].  Returns None if no boxes.

    Mirrors the core math in inference/run.py:_compute_player_crop.
    """
    if not boxes:
        return None

    px1, py1, px2, py2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    ph  = py2 - py1
    pcx = (px1 + px2) / 2.0

    crop_h = ph / _PLAYER_FILL
    crop_w = crop_h * (2.0 / 3.0)

    crop_y1 = py1 - _HEAD_ROOM * crop_h
    crop_x1 = pcx - crop_w / 2.0

    crop_x1 = max(0.0, min(crop_x1, W - crop_w))
    crop_y1 = max(0.0, min(crop_y1, H - crop_h))
    crop_w  = min(crop_w, W - crop_x1)
    crop_h  = min(crop_h, H - crop_y1)

    return [
        crop_x1 / W,
        crop_y1 / H,
        (crop_x1 + crop_w) / W,
        (crop_y1 + crop_h) / H,
    ]


def build(fallback_skip: bool = False) -> None:
    from ultralytics import YOLO

    if not CROP_GT_FILE.exists():
        raise FileNotFoundError(
            f"Run 'python -m data.crop_detector' first — {CROP_GT_FILE} not found."
        )

    with open(CROP_GT_FILE) as fh:
        records = json.load(fh)

    bundled  = CHECKPOINTS_DIR / "yolo11n.pt"
    weights  = str(bundled) if bundled.exists() else "yolo11n.pt"
    detector = YOLO(weights)

    out_records: list[dict] = []
    n_detected = n_fallback = n_skipped = 0

    for rec in tqdm(records, desc="Building YOLO crop GT"):
        try:
            thumb = extract_thumbnail_ar(rec["raw"], max_size=_EXTRACT_SIZE)
        except Exception:
            n_skipped += 1
            continue

        W, H     = thumb.size
        min_area = W * H * _MIN_FRACTION

        try:
            results = detector(thumb, classes=[0], verbose=False)
            boxes = [
                (x1, y1, x2, y2)
                for r in results
                for (x1, y1, x2, y2) in (map(float, b) for b in r.boxes.xyxy.cpu().tolist())
                if (x2 - x1) * (y2 - y1) >= min_area
            ]
        except Exception:
            boxes = []

        yolo_box = _player_crop_normalized(boxes, W, H)

        if yolo_box is not None:
            n_detected += 1
            out_records.append({
                "box":           yolo_box,
                "angle_deg":     0.0,
                "split":         rec["split"],
                "raw":           rec["raw"],
                "edited":        rec.get("edited", ""),
                "yolo_detected": True,
            })
        elif fallback_skip:
            n_skipped += 1
        else:
            n_fallback += 1
            out_records.append({
                "box":           rec["box"],
                "angle_deg":     rec.get("angle_deg", 0.0),
                "split":         rec["split"],
                "raw":           rec["raw"],
                "edited":        rec.get("edited", ""),
                "yolo_detected": False,
            })

    CROP_GT_YOLO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CROP_GT_YOLO_FILE, "w") as fh:
        json.dump(out_records, fh)

    total = n_detected + n_fallback + n_skipped
    print(f"\n{'='*55}")
    print(f"  Raws processed      : {total:,}")
    print(f"  YOLO detected       : {n_detected:,}  ({n_detected/max(total,1):.1%})")
    if not fallback_skip:
        print(f"  Fallback (orig GT)  : {n_fallback:,}  ({n_fallback/max(total,1):.1%})")
    print(f"  Skipped (read error): {n_skipped:,}")
    print(f"  Output              : {CROP_GT_YOLO_FILE}")
    print(f"  Records written     : {len(out_records):,}")
    print("=" * 55)

    split_counts = {}
    for r in out_records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    for split, cnt in sorted(split_counts.items()):
        print(f"    {split:6s}: {cnt:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fallback-skip", action="store_true",
        help="Drop raws where YOLO detects no player (don't fall back to original GT)",
    )
    args = parser.parse_args()
    build(fallback_skip=args.fallback_skip)
