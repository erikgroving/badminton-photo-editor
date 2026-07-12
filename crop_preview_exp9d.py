"""
Render randomly sampled test-set photos comparing Jay's GT crop with the
exp9d model's prediction — three panels per photo, all properly oriented:

  [ original + both boxes ]  [ Jay's crop ]  [ exp9d's crop ]
     green = Jay, orange = exp9d

Orientation rules (avoids upside-down humans):
  - The overlay is drawn in sensor-native space, then rotated by the raw's
    own orientation flag (flip), so the rectangles ride along.
  - Crops from orientation-flagged raws (flip 3/5/6) inherit the photo's
    upright rotation.
  - Crops from landscape raws that Jay rotated to portrait (angle >= 45)
    pick CW vs CCW using pose keypoints: heads go above ankles.
    Without usable pose, falls back to CW (the apply_crop convention).

Runs on CPU by design (thread-capped) so it can run alongside GPU training.

Usage:
    python crop_preview_exp9d.py              # 200 samples -> crop_preview_exp9d.html
    python crop_preview_exp9d.py --n 50
    python crop_preview_exp9d.py --no-open
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import time
import webbrowser
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_DIR, CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar, get_raw_flip, _apply_flip
from models.cropping.model import build_exp9_model, compute_region_stats, POSE_DIM

_CKPT  = CHECKPOINTS_DIR / "cropping_angle_vit_large_patch14_reg4_dinov2_exp9d_full.pt"
_MEAN  = (0.485, 0.456, 0.406)
_STD   = (0.229, 0.224, 0.225)
_THUMB   = 512        # model-input thumbnail (matches training/inference)
_DISPLAY = 656        # larger thumbnail for the page panels
_POSE_CONF_THRESH = 0.3

# COCO keypoint indices
_NOSE, _L_EYE, _R_EYE = 0, 1, 2
_L_ANKLE, _R_ANKLE    = 15, 16


def _load_model():
    ck = torch.load(_CKPT, map_location="cpu", weights_only=False)
    model = build_exp9_model(
        pretrained=False,
        pose_dim=ck.get("pose_dim", POSE_DIM),
        cond_emb_dim=ck.get("cond_emb_dim", 128),
        dynamic_img_size=ck.get("dynamic_img_size", True),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    m = ck.get("metrics", {})
    print(f"Loaded exp9d ep{ck.get('epoch', '?')}: "
          f"val_iou={m.get('mean_iou', 0):.4f}  angle_mae={m.get('angle_mae_deg', 0):.2f}deg")
    return model, int(ck.get("input_size", 770))


def _pose_from_cache(raw_kpts):
    """[x,y,conf]x17 -> 34 floats, zeroing low-confidence keypoints (training convention)."""
    if raw_kpts is None:
        return [0.0] * POSE_DIM
    kpts = []
    for i in range(POSE_DIM // 2):
        x, y, c = raw_kpts[i*3], raw_kpts[i*3+1], raw_kpts[i*3+2]
        kpts.extend([x, y] if c >= _POSE_CONF_THRESH else [0.0, 0.0])
    return kpts


def _portrait_rotation(raw_kpts) -> int:
    """For a landscape-shot raw whose crop is rotated to portrait: choose the
    rotation that puts the head above the ankles.
    Returns PIL rotate degrees: -90 (CW, default) or 90 (CCW).
    CW  maps sensor-native (x, y) -> (1-y, x); CCW maps (x, y) -> (y, 1-x)."""
    if raw_kpts is None:
        return -90
    def pt(i):
        x, y, c = raw_kpts[i*3], raw_kpts[i*3+1], raw_kpts[i*3+2]
        return (x, y) if c >= _POSE_CONF_THRESH else None
    head   = next((p for p in (pt(_NOSE), pt(_L_EYE), pt(_R_EYE)) if p), None)
    ankles = [p for p in (pt(_L_ANKLE), pt(_R_ANKLE)) if p]
    if head is None or not ankles:
        return -90
    ax = sum(a[0] for a in ankles) / len(ankles)
    # CW: display head_y = head_x, ankle_y = ankle_x -> head above ankles iff head_x < ankle_x
    return -90 if head[0] < ax else 90


@torch.no_grad()
def _predict(model, input_size, thumb, union_bbox, primary_bbox, kpts):
    """Returns (box [x1,y1,x2,y2] in [0,1], angle_deg) — soft blend, same as val."""
    t = TF.to_tensor(thumb.resize((input_size, input_size), Image.LANCZOS))
    img_t = TF.normalize(t, _MEAN, _STD).unsqueeze(0)
    ub    = torch.tensor([union_bbox],   dtype=torch.float32)
    pb    = torch.tensor([primary_bbox], dtype=torch.float32)
    st    = torch.tensor([compute_region_stats(thumb)], dtype=torch.float32)
    kp    = torch.tensor([kpts], dtype=torch.float32)
    box, angle = model(img_t, ub, pb, st, kp)
    box = [max(0.0, min(1.0, v)) for v in box[0].tolist()]
    return box, float(angle.item()) * 90.0


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0]) * (a[3]-a[1])
    ub = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(ua + ub - inter, 1e-6)


def _draw_boxes(img, gt, pred):
    out  = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    px = lambda b: [b[0]*w, b[1]*h, b[2]*w, b[3]*h]
    draw.rectangle(px(gt),   outline=(40, 220, 40),  width=3)
    draw.rectangle(px(pred), outline=(255, 100, 20), width=3)
    return out


def _crop_view(thumb, box, angle_deg, flip, raw_kpts):
    """Crop the box region from the sensor-native thumb and rotate it upright."""
    w, h = thumb.size
    x1, y1 = max(0, int(box[0]*w)), max(0, int(box[1]*h))
    x2, y2 = min(w, int(box[2]*w)), min(h, int(box[3]*h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return Image.new("RGB", (60, 40), (30, 30, 30))
    out = thumb.crop((x1, y1, x2, y2))
    if flip in (3, 5, 6):
        # Photo itself was shot rotated — the crop inherits the photo's upright rotation
        return _apply_flip(out, flip)
    if angle_deg >= 45.0:
        # Landscape shot, Jay rotated the crop to portrait — pose decides direction
        return out.rotate(_portrait_rotation(raw_kpts), expand=True)
    return out


def _data_url(img, quality=74):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=200)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    # CPU only, and leave cores for the GPU training's dataloader workers
    n_threads = max(2, (os.cpu_count() or 8) // 2)
    torch.set_num_threads(n_threads)
    print(f"CPU inference, {n_threads} torch threads")

    with open(CROP_GT_FILE) as f:
        all_gt = json.load(f)
    test_gt = [e for e in all_gt if e.get("split") == "test"]

    def _cache(name):
        p = BASE_DIR / "data" / name
        return json.load(open(p)) if p.exists() else {}
    union_cache   = _cache("player_bboxes.json")
    primary_cache = _cache("primary_player_bboxes.json")
    pose_cache    = _cache("pose_keypoints.json")

    rng    = random.Random(args.seed)
    sample = rng.sample(test_gt, min(args.n, len(test_gt)))
    print(f"Sampling {len(sample)} of {len(test_gt)} test photos")

    model, input_size = _load_model()

    cards, t0 = [], time.time()
    for i, entry in enumerate(sample):
        raw = entry["raw"]
        gt  = entry["box"]
        try:
            thumb    = extract_thumbnail_ar(raw, max_size=_THUMB)    # model input
            disp     = extract_thumbnail_ar(raw, max_size=_DISPLAY)  # page panels
            flip     = get_raw_flip(raw)
            raw_kpts = pose_cache.get(raw)
            ub       = union_cache.get(raw, [0.0, 0.0, 0.0, 0.0])
            pb       = primary_cache.get(raw, ub)
            kp       = _pose_from_cache(raw_kpts)

            pred, angle_pred = _predict(model, input_size, thumb, ub, pb, kp)
            score    = _iou(gt, pred)
            angle_gt = entry.get("angle_deg", 0.0)

            # Overlay: draw in sensor-native space, then orient the whole image
            vis = _apply_flip(_draw_boxes(disp, gt, pred), flip)

            cards.append({
                "name":       Path(raw).name,
                "iou":        score,
                "angle_gt":   angle_gt,
                "angle_pred": angle_pred,
                "overlay":    _data_url(vis),
                "crop_gt":    _data_url(_crop_view(disp, gt, angle_gt, flip, raw_kpts)),
                "crop_pred":  _data_url(_crop_view(disp, pred, angle_pred, flip, raw_kpts)),
            })
            el  = time.time() - t0
            eta = el / (i + 1) * (len(sample) - i - 1)
            print(f"  [{i+1:3d}/{len(sample)}] {Path(raw).name}  iou={score:.3f}  "
                  f"(eta {eta/60:.0f}m)", flush=True)
        except Exception as exc:
            print(f"  [{i+1:3d}/{len(sample)}] SKIP {Path(raw).name} ({exc})", flush=True)

    cards.sort(key=lambda c: c["name"])
    mean_iou = sum(c["iou"] for c in cards) / max(len(cards), 1)
    print(f"\nSample mean IoU: {mean_iou:.4f}  |  n={len(cards)}")

    out = BASE_DIR / "crop_preview_exp9d.html"
    out.write_text(_render_html(cards, mean_iou), encoding="utf-8")
    print(f"Saved: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


def _render_html(cards, mean_iou):
    items = ""
    for c in cards:
        color = "#2ecc71" if c["iou"] >= 0.8 else ("#f59e0b" if c["iou"] >= 0.6 else "#e74c3c")
        items += f"""
  <div class="card">
    <div class="row">
      <div class="cell main"><img src="{c['overlay']}" loading="lazy"></div>
      <div class="cell"><div class="tag jay">Jay</div><img src="{c['crop_gt']}" loading="lazy"></div>
      <div class="cell"><div class="tag mdl">exp9d</div><img src="{c['crop_pred']}" loading="lazy"></div>
    </div>
    <div class="meta">
      <span class="fn">{c['name']}</span>
      <span class="iou" style="color:{color}">IoU {c['iou']:.3f}</span>
      <span class="ang">&ang; jay {c['angle_gt']:.1f}&deg; / model {c['angle_pred']:.1f}&deg;</span>
    </div>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>exp9d Crop Preview</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#ddd;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
  #hdr{{background:#0a0a0a;border-bottom:1px solid #222;padding:10px 18px;
        display:flex;align-items:center;gap:18px;position:sticky;top:0;z-index:10}}
  #hdr h1{{font-size:14px;font-weight:600;color:#fff;white-space:nowrap}}
  .stat{{font-size:12px;color:#888}}
  .legend{{font-size:12px;white-space:nowrap}}
  .list{{display:flex;flex-direction:column;gap:10px;padding:12px;max-width:1600px;margin:0 auto}}
  .card{{background:#1a1a1a;border-radius:6px;overflow:hidden;border:1px solid #252525}}
  .row{{display:flex;gap:6px;padding:6px;align-items:stretch}}
  .cell{{position:relative;flex:1;min-width:0;display:flex;align-items:center;justify-content:center;background:#141414;border-radius:4px;overflow:hidden}}
  .cell.main{{flex:1.6}}
  .cell img{{max-width:100%;max-height:480px;display:block;object-fit:contain}}
  .tag{{position:absolute;top:6px;left:6px;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;z-index:2}}
  .tag.jay{{background:#28dc28;color:#052905}}
  .tag.mdl{{background:#ff6414;color:#2b1000}}
  .meta{{padding:6px 10px;display:flex;align-items:center;gap:10px;border-top:1px solid #222}}
  .fn{{font-size:11px;color:#666;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .iou{{font-size:13px;font-weight:700;white-space:nowrap}}
  .ang{{font-size:11px;color:#555;white-space:nowrap}}
</style>
</head>
<body>
<div id="hdr">
  <h1>exp9d Crop Preview — {len(cards)} test photos</h1>
  <span class="stat">Sample mean IoU <b style="color:#fff">{mean_iou:.4f}</b></span>
  <span class="legend">
    <span style="color:#28dc28">&#9646;</span> Jay &nbsp;
    <span style="color:#ff6414">&#9646;</span> exp9d
  </span>
  <span class="stat" style="margin-left:auto">sorted by filename</span>
</div>
<div class="list">{items}
</div>
</body>
</html>"""


if __name__ == "__main__":
    main()
