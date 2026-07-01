"""
E2E crop diagnostic: checks orientation, image sizes, and model predictions
vs GT boxes for a sample of test images.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import CROP_GT_FILE, CHECKPOINTS_DIR, DEVELOP_SIZE
from data.raw_reader import develop_raw, extract_thumbnail_ar, get_raw_flip
from inference.pipeline import _load_crop, predict_crop, _developed_image_box

DIAG_DIR = ROOT / "diag_crops"
DIAG_DIR.mkdir(exist_ok=True)

with open(CROP_GT_FILE) as f:
    recs = json.load(f)
test_recs = [r for r in recs if r.get("split") == "test"][:12]

# ── 1. Orientation check ─────────────────────────────────────────────────────
print("\n=== ORIENTATION CHECK ===")
seen_flips = {}
for r in recs:
    flip = get_raw_flip(r["raw"])
    if flip not in seen_flips:
        seen_flips[flip] = r
for flip, r in sorted(seen_flips.items()):
    thumb   = extract_thumbnail_ar(r["raw"], max_size=512)
    raw_img = develop_raw(r["raw"], size=DEVELOP_SIZE, neutral=False)
    print(f"flip={flip}  thumb={thumb.size}  develop={raw_img.size}  gt={r['box'][:2]}")

# ── 2. Load model ─────────────────────────────────────────────────────────────
print("\n=== MODEL ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crop_model = _load_crop(device)
print(f"Model type: {type(crop_model).__name__}")
print(f"Input size: {getattr(crop_model, '_input_size', '?')}")

# ── 3. Run predict_crop on test images ───────────────────────────────────────
print("\n=== PREDICTIONS vs GT ===")
from models.cropping.model import box_iou_numpy
import numpy as np

ious = []
for r in test_recs:
    raw_img = develop_raw(r["raw"], size=DEVELOP_SIZE, neutral=False)
    flip    = get_raw_flip(r["raw"])

    crop_box, angle = predict_crop(crop_model, r["raw"], device,
                                   img_size=raw_img.size)

    # pred_norm: in developed-image (correctly oriented) space
    W, H = raw_img.size
    px, py, pw, ph = crop_box
    pred_norm = [px/W, py/H, (px+pw)/W, (py+ph)/H]

    # GT is in sensor-native (thumbnail/landscape) space; convert to developed space
    gt_sn  = r["box"]
    gt_dev = _developed_image_box(gt_sn, flip)

    iou = float(box_iou_numpy(np.array([pred_norm]), np.array([gt_dev]))[0])
    ious.append(iou)
    stem = Path(r["raw"]).stem

    print(f"  {stem[:30]}  flip={flip}  W={W} H={H}")
    print(f"    gt_sn  = [{gt_sn[0]:.3f},{gt_sn[1]:.3f},{gt_sn[2]:.3f},{gt_sn[3]:.3f}]")
    print(f"    gt_dev = [{gt_dev[0]:.3f},{gt_dev[1]:.3f},{gt_dev[2]:.3f},{gt_dev[3]:.3f}]")
    print(f"    pred   = [{pred_norm[0]:.3f},{pred_norm[1]:.3f},{pred_norm[2]:.3f},{pred_norm[3]:.3f}]  angle={angle:.1f}  IoU={iou:.3f}")

    # Save visual overlay on developed image (both boxes in same space)
    vis = raw_img.copy().resize((800, int(800 * H / W)), Image.LANCZOS)
    vW, vH = vis.size
    draw = ImageDraw.Draw(vis)
    # GT in green (converted to developed-image space)
    draw.rectangle([gt_dev[0]*vW, gt_dev[1]*vH, gt_dev[2]*vW, gt_dev[3]*vH],
                   outline=(0,255,0), width=3)
    # Pred in red
    draw.rectangle([pred_norm[0]*vW, pred_norm[1]*vH, pred_norm[2]*vW, pred_norm[3]*vH],
                   outline=(255,0,0), width=3)
    vis.save(DIAG_DIR / f"{stem}_iou{iou:.2f}.jpg", quality=88)

print(f"\nmIoU = {np.mean(ious):.4f}  median = {np.median(ious):.4f}")
print(f"Saved overlays to {DIAG_DIR}")
