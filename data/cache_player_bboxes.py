"""
Pre-compute player bounding boxes for every raw file in color_gt.json and
crop_gt.json.  Two caches are produced:

  data/player_bboxes.json         — union of all detected persons (for model conditioning)
  data/primary_player_bboxes.json — largest detected person only (for clipping penalty)

Format: { raw_path: [x1_norm, y1_norm, x2_norm, y2_norm] }
  Coords are normalised to [0, 1] relative to the thumbnail dimensions.

Run once before training:
    python data/cache_player_bboxes.py

Subsequent runs skip already-cached paths (incremental).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_GT_FILE, CROP_GT_FILE, COLOR_SIZE, CHECKPOINTS_DIR
from data.raw_reader import extract_thumbnail

_CACHE_FILE         = Path(__file__).parent / "player_bboxes.json"
_PRIMARY_CACHE_FILE = Path(__file__).parent / "primary_player_bboxes.json"
_MIN_CROP_FRACTION  = 0.01


def build_detector():
    from ultralytics import YOLO
    bundled = CHECKPOINTS_DIR / "yolo11n.pt"
    weights = str(bundled) if bundled.exists() else "yolo11n.pt"
    return YOLO(weights)


def _detect_boxes(img, detector):
    """Return list of (x1, y1, x2, y2) pixel coords for all detected persons."""
    w, h = img.size
    min_area = w * h * _MIN_CROP_FRACTION
    results = detector(img, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(float, box)
            if (x2 - x1) * (y2 - y1) >= min_area:
                boxes.append((x1, y1, x2, y2))
    return boxes


def detect_union_bbox(img, detector):
    """Return union bbox [x1n, y1n, x2n, y2n] of all person detections, or None."""
    w, h = img.size
    boxes = _detect_boxes(img, detector)
    if not boxes:
        return None, None
    ux1 = min(b[0] for b in boxes)
    uy1 = min(b[1] for b in boxes)
    ux2 = max(b[2] for b in boxes)
    uy2 = max(b[3] for b in boxes)
    px1, py1, px2, py2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    union   = [ux1 / w, uy1 / h, ux2 / w, uy2 / h]
    primary = [px1 / w, py1 / h, px2 / w, py2 / h]
    return union, primary


def main():
    raw_paths: set[str] = set()
    for gt_file in (COLOR_GT_FILE, CROP_GT_FILE):
        if gt_file.exists():
            records = json.load(open(gt_file))
            raw_paths.update(r["raw"] for r in records)
        else:
            print(f"  (skipping {gt_file.name} — not found)")
    if not raw_paths:
        print("ERROR: no GT files found. Run data/xmp_reader.py or data/crop_detector.py first.")
        sys.exit(1)
    raw_paths_list = list(raw_paths)
    print(f"Total unique raw files (color + crop GT): {len(raw_paths_list)}")

    union_cache: dict = {}
    if _CACHE_FILE.exists():
        union_cache = json.load(open(_CACHE_FILE))
        print(f"Loaded {len(union_cache)} union cached entries")

    primary_cache: dict = {}
    if _PRIMARY_CACHE_FILE.exists():
        primary_cache = json.load(open(_PRIMARY_CACHE_FILE))
        print(f"Loaded {len(primary_cache)} primary cached entries")

    # Process paths missing from either cache
    missing = [p for p in raw_paths_list
               if p not in union_cache or p not in primary_cache]
    print(f"Need to process: {len(missing)}")

    if not missing:
        print("All entries cached. Nothing to do.")
        return

    detector = build_detector()

    hits = 0
    for i, raw_path in enumerate(missing):
        if not Path(raw_path).exists():
            union_cache[raw_path]   = None
            primary_cache[raw_path] = None
            continue
        try:
            img             = extract_thumbnail(raw_path, size=COLOR_SIZE)
            union, primary  = detect_union_bbox(img, detector)
            union_cache[raw_path]   = union
            primary_cache[raw_path] = primary
            if union:
                hits += 1
        except Exception as e:
            print(f"  Error on {Path(raw_path).name}: {e}")
            union_cache[raw_path]   = None
            primary_cache[raw_path] = None

        if (i + 1) % 50 == 0 or (i + 1) == len(missing):
            json.dump(union_cache,   open(_CACHE_FILE,         "w"), indent=2)
            json.dump(primary_cache, open(_PRIMARY_CACHE_FILE, "w"), indent=2)
            pct = (i + 1) / len(missing) * 100
            print(f"  [{i+1}/{len(missing)}] {pct:.0f}%  players found: {hits}")

    json.dump(union_cache,   open(_CACHE_FILE,         "w"), indent=2)
    json.dump(primary_cache, open(_PRIMARY_CACHE_FILE, "w"), indent=2)
    found    = sum(1 for v in union_cache.values() if v is not None)
    no_found = sum(1 for v in union_cache.values() if v is None)
    print(f"\nDone. {found} with players / {no_found} without.")
    print(f"  Union cache  → {_CACHE_FILE}")
    print(f"  Primary cache→ {_PRIMARY_CACHE_FILE}")


if __name__ == "__main__":
    main()
