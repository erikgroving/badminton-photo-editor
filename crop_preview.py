"""
Render 100 randomly sampled test-set photos with crop boxes overlaid:
  Green  = Jay's ground truth crop
  Orange = model prediction

Usage:
    python crop_preview.py            # 100 samples, opens browser
    python crop_preview.py --n 50     # fewer samples
    python crop_preview.py --no-open  # don't auto-open
"""
import argparse
import base64
import io
import json
import random
import sys
import webbrowser
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from inference.pipeline import _load_crop
from models.cropping.model import CombinedDINOv2, compute_region_stats

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)
_DISPLAY_MAX = 700   # longest edge of the thumbnail shown in the page


def _to_tensor(img: Image.Image, size: int, device) -> torch.Tensor:
    t = TF.to_tensor(img.resize((size, size), Image.LANCZOS))
    return TF.normalize(t, _MEAN, _STD).unsqueeze(0).to(device)


@torch.no_grad()
def _predict(model, thumb: Image.Image, device,
             union_bbox, primary_bbox) -> list[float]:
    """Return predicted box as [x1, y1, x2, y2] in [0, 1]."""
    sz = getattr(model, '_input_size', 518)
    img_t = _to_tensor(thumb, sz, device)

    if isinstance(model, CombinedDINOv2):
        stats = torch.tensor([compute_region_stats(thumb)],
                             dtype=torch.float32, device=device)
        ub    = torch.tensor([union_bbox],   dtype=torch.float32, device=device)
        pb    = torch.tensor([primary_bbox], dtype=torch.float32, device=device)
        box, _ = model(img_t, ub, pb, stats)
    else:
        out = model(img_t)
        box = out[0] if isinstance(out, tuple) else out

    return [max(0.0, min(1.0, v)) for v in box[0].cpu().tolist()]


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0]) * (a[3]-a[1])
    ub = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(ua + ub - inter, 1e-6)


def _draw_boxes(img: Image.Image, gt: list, pred: list) -> Image.Image:
    out  = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    def px(box):
        return [box[0]*w, box[1]*h, box[2]*w, box[3]*h]

    draw.rectangle(px(gt),   outline=(40, 220, 40),  width=3)
    draw.rectangle(px(pred), outline=(255, 100, 20), width=3)
    return out


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=200)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ground truth
    with open(CROP_GT_FILE) as f:
        all_gt = json.load(f)
    test_gt = [e for e in all_gt if e.get("split") == "test"]
    print(f"Test entries in crop_gt: {len(test_gt)}")

    # Load pre-computed player bbox caches (produced by data/cache_player_bboxes.py)
    def _load_cache(path):
        p = BASE_DIR / "data" / path
        return json.load(open(p)) if p.exists() else {}

    union_cache   = _load_cache("player_bboxes.json")
    primary_cache = _load_cache("primary_player_bboxes.json")
    if not union_cache:
        print("Note: player_bboxes.json not found — model will run without player conditioning")

    # Sample
    rng    = random.Random(args.seed)
    sample = rng.sample(test_gt, min(args.n, len(test_gt)))
    print(f"Sampling {len(sample)} photos…")

    # Load model
    model = _load_crop(device)

    # Run inference
    cards = []
    for i, entry in enumerate(sample):
        raw = entry["raw"]
        gt  = entry["box"]  # [x1, y1, x2, y2] in [0, 1]

        print(f"  [{i+1:3d}/{len(sample)}] {Path(raw).name}", end="", flush=True)
        try:
            thumb = extract_thumbnail_ar(raw, max_size=_DISPLAY_MAX)
            ub = union_cache.get(raw,   [0.0, 0.0, 0.0, 0.0])
            pb = primary_cache.get(raw, ub)

            pred  = _predict(model, thumb, device, ub, pb)
            score = _iou(gt, pred)
            print(f"  iou={score:.3f}")

            vis = _draw_boxes(thumb, gt, pred)
            cards.append({
                "name":      Path(raw).name,
                "iou":       score,
                "angle_gt":  entry.get("angle_deg", 0.0),
                "data_url":  _data_url(vis),
            })
        except Exception as exc:
            print(f"  SKIP ({exc})")

    cards.sort(key=lambda c: c["name"])
    mean_iou = sum(c["iou"] for c in cards) / max(len(cards), 1)
    print(f"\nMean IoU (sample): {mean_iou:.3f}  |  n={len(cards)}")

    html = _render_html(cards, mean_iou)
    out  = BASE_DIR / "crop_preview.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")

    if not args.no_open:
        webbrowser.open(out.as_uri())


def _render_html(cards, mean_iou):
    items = ""
    for c in cards:
        color = "#2ecc71" if c["iou"] >= 0.7 else ("#f59e0b" if c["iou"] >= 0.5 else "#e74c3c")
        items += f"""
  <div class="card">
    <img src="{c['data_url']}" loading="lazy">
    <div class="meta">
      <span class="fn">{c['name']}</span>
      <span class="iou" style="color:{color}">IoU {c['iou']:.3f}</span>
      <span class="ang">∠{c['angle_gt']:.1f}°</span>
    </div>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Crop Preview</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#ddd;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
  #hdr{{background:#0a0a0a;border-bottom:1px solid #222;padding:10px 18px;
        display:flex;align-items:center;gap:18px;position:sticky;top:0;z-index:10}}
  #hdr h1{{font-size:14px;font-weight:600;color:#fff;white-space:nowrap}}
  .stat{{font-size:12px;color:#888}}
  .legend{{font-size:12px;white-space:nowrap}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:12px}}
  .card{{background:#1a1a1a;border-radius:5px;overflow:hidden;border:1px solid #252525}}
  .card img{{width:100%;display:block}}
  .meta{{padding:5px 9px;display:flex;align-items:center;gap:8px}}
  .fn{{font-size:10px;color:#666;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .iou{{font-size:13px;font-weight:700;white-space:nowrap}}
  .ang{{font-size:10px;color:#555;white-space:nowrap}}
</style>
</head>
<body>
<div id="hdr">
  <h1>Crop Preview — {len(cards)} test photos</h1>
  <span class="stat">Mean IoU <b style="color:#fff">{mean_iou:.3f}</b></span>
  <span class="legend">
    <span style="color:#28dc28">&#9646;</span> Jay &nbsp;
    <span style="color:#ff6414">&#9646;</span> Model
  </span>
  <span class="stat" style="margin-left:auto">sorted by filename</span>
</div>
<div class="grid">{items}
</div>
</body>
</html>"""


if __name__ == "__main__":
    main()
