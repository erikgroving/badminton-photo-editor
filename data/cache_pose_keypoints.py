"""
Pre-compute pose keypoints for every raw file in crop_gt.json.

Uses YOLO11n-pose to detect the primary player's 17 COCO keypoints.
If pose detection fails or yields no person, falls back to approximating
keypoints from the primary player bbox.

Cache: data/pose_keypoints.json
Format: { raw_path: [kp1_x, kp1_y, kp1_conf, kp2_x, ...] }   (51 floats, normalized [0,1])

COCO keypoint order (17 points):
  0: nose          1: left_eye      2: right_eye     3: left_ear    4: right_ear
  5: left_shoulder 6: right_shoulder 7: left_elbow   8: right_elbow 9: left_wrist
 10: right_wrist  11: left_hip     12: right_hip    13: left_knee  14: right_knee
 15: left_ankle   16: right_ankle

Run once before training:
    python data/cache_pose_keypoints.py

Subsequent runs skip already-cached paths (incremental).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROP_GT_FILE, COLOR_GT_FILE, CHECKPOINTS_DIR
from data.raw_reader import extract_thumbnail

_CACHE_FILE = Path(__file__).parent / "pose_keypoints.json"
_POSE_MODEL_PATH = Path("yolo11n-pose.pt")  # will be in cwd or auto-downloaded
_MIN_CROP_FRACTION = 0.01

# COCO keypoint skeleton — used for bbox-fallback approximation
# Normalized offsets from bbox center for each of the 17 keypoints.
# Format: (dx_frac_of_w, dy_frac_of_h) relative to bbox center.
# This is a rough anatomical prior for a standing badminton player.
_COCO_APPROX_OFFSETS = [
    (0.00, -0.45),   # 0: nose         — top-center, near head top
    (-0.05, -0.47),  # 1: left_eye
    (0.05, -0.47),   # 2: right_eye
    (-0.08, -0.46),  # 3: left_ear
    (0.08, -0.46),   # 4: right_ear
    (-0.15, -0.30),  # 5: left_shoulder
    (0.15, -0.30),   # 6: right_shoulder
    (-0.22, -0.08),  # 7: left_elbow
    (0.22, -0.08),   # 8: right_elbow
    (-0.25,  0.10),  # 9: left_wrist
    (0.25,  0.10),   # 10: right_wrist
    (-0.12,  0.08),  # 11: left_hip
    (0.12,  0.08),   # 12: right_hip
    (-0.13,  0.30),  # 13: left_knee
    (0.13,  0.30),   # 14: right_knee
    (-0.12,  0.45),  # 15: left_ankle
    (0.12,  0.45),   # 16: right_ankle
]


def _bbox_approx_keypoints(bbox_norm: list[float]) -> list[float]:
    """
    Approximate 17 COCO keypoints from a normalized bbox [x1, y1, x2, y2].
    Returns flat list of 51 floats: [x, y, conf] * 17, all in [0, 1].
    Confidence set to 0.5 (medium, since these are approximated).
    """
    x1, y1, x2, y2 = bbox_norm
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1

    kps = []
    for dx, dy in _COCO_APPROX_OFFSETS:
        kx = cx + dx * w
        ky = cy + dy * h
        kps.extend([kx, ky, 0.5])  # conf=0.5 signals "approximated"
    return kps


def build_pose_detector():
    """Load YOLO pose model, preferring bundled copy if available.
    We pass device='cpu' in each inference call to avoid CUDA OOM when
    processing ~3700 images sequentially (YOLO11n-pose is tiny, ~0.06s/img on CPU).
    Do NOT call model.to('cpu') here — it triggers a GPU probe in ultralytics
    that can fail in subprocess contexts. Just pass device='cpu' at inference time.
    """
    from ultralytics import YOLO
    bundled = CHECKPOINTS_DIR / "yolo11n-pose.pt"
    if bundled.exists():
        weights = str(bundled)
    elif _POSE_MODEL_PATH.exists():
        weights = str(_POSE_MODEL_PATH)
    else:
        weights = "yolo11n-pose.pt"  # triggers auto-download
    return YOLO(weights)


def detect_primary_pose(img, detector) -> list[float] | None:
    """
    Run YOLO11n-pose on `img`, return flat 51-float keypoint list for the
    primary (largest area) person, or None if no person detected.
    Coordinates are normalized to [0, 1].
    """
    w, h = img.size
    min_area = w * h * _MIN_CROP_FRACTION

    results = detector(img, classes=[0], verbose=False, device="cpu")
    best_kps = None
    best_area = -1.0

    for r in results:
        if r.keypoints is None:
            continue
        boxes = r.boxes.xyxy.cpu().tolist()        # [N, 4]
        kps_all = r.keypoints.data.cpu().tolist()  # [N, 17, 3]

        for box, kps in zip(boxes, kps_all):
            x1, y1, x2, y2 = map(float, box)
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            if area > best_area:
                best_area = area
                best_kps = kps

    if best_kps is None:
        return None

    # Flatten to 51-float list, normalize to [0, 1]
    flat = []
    for kx, ky, kconf in best_kps:
        flat.extend([kx / w, ky / h, float(kconf)])
    return flat


def main():
    # Collect all raw paths from crop_gt.json (and optionally color_gt.json)
    raw_paths: set[str] = set()
    for gt_file in (CROP_GT_FILE,):
        if gt_file.exists():
            records = json.load(open(gt_file))
            raw_paths.update(r["raw"] for r in records)
        else:
            print(f"  (skipping {gt_file.name} — not found)")

    if not raw_paths:
        print("ERROR: no GT files found. Run data/crop_detector.py first.")
        sys.exit(1)

    raw_paths_list = list(raw_paths)
    print(f"Total unique raw files: {len(raw_paths_list)}")

    # Load existing cache (incremental)
    cache: dict = {}
    if _CACHE_FILE.exists():
        cache = json.load(open(_CACHE_FILE))
        print(f"Loaded {len(cache)} cached entries")

    missing = [p for p in raw_paths_list if p not in cache]
    print(f"Need to process: {len(missing)}")

    if not missing:
        print("All entries cached. Nothing to do.")
        return

    # Load primary bbox cache for fallback approximation
    primary_cache_path = Path(__file__).parent / "primary_player_bboxes.json"
    primary_cache: dict = {}
    if primary_cache_path.exists():
        primary_cache = json.load(open(primary_cache_path))
        print(f"  Primary bbox cache loaded ({len(primary_cache)} entries) — used as fallback")
    else:
        print("  Warning: no primary_player_bboxes.json — bbox fallback unavailable")

    detector = build_pose_detector()

    pose_hits  = 0   # detected via YOLO pose
    bbox_hits  = 0   # approximated from primary bbox
    miss_count = 0   # neither available

    for i, raw_path in enumerate(missing):
        if not Path(raw_path).exists():
            cache[raw_path] = None
            miss_count += 1
            continue

        kps = None
        try:
            img = extract_thumbnail(raw_path, size=(512, 512))
            kps = detect_primary_pose(img, detector)
            if kps is not None:
                pose_hits += 1
        except Exception as e:
            print(f"  Pose error on {Path(raw_path).name}: {e}")

        if kps is None:
            # Fallback: approximate from primary bbox
            bbox = primary_cache.get(raw_path)
            if bbox is not None:
                kps = _bbox_approx_keypoints(bbox)
                bbox_hits += 1
            else:
                miss_count += 1

        cache[raw_path] = kps

        if (i + 1) % 50 == 0 or (i + 1) == len(missing):
            json.dump(cache, open(_CACHE_FILE, "w"), indent=2)
            pct = (i + 1) / len(missing) * 100
            print(f"  [{i+1}/{len(missing)}] {pct:.0f}%  "
                  f"pose={pose_hits}  bbox_approx={bbox_hits}  miss={miss_count}")

    # Final save
    json.dump(cache, open(_CACHE_FILE, "w"), indent=2)
    total_valid = sum(1 for v in cache.values() if v is not None)
    print(f"\nDone.")
    print(f"  pose detection: {pose_hits}")
    print(f"  bbox approximation: {bbox_hits}")
    print(f"  no keypoints: {miss_count}")
    print(f"  total valid: {total_valid}/{len(cache)}")
    print(f"  Cache → {_CACHE_FILE}")


if __name__ == "__main__":
    main()
