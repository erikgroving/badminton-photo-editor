"""
Visual QA panel for the color-affine model: [ before | AI color | Jay's edit ]
per photo with per-photo dE2000, on random val pairs.

Usage:
    python color_preview.py             # 90 samples -> color_preview.html
    python color_preview.py --n 40
"""
import argparse
import base64
import io
import sys
import webbrowser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_DIR, CHECKPOINTS_DIR, COLOR_SIZE, WATERMARK_REGION
from data.raw_reader import develop_raw
from eval_color_imagespace import srgb_to_lab, de2000
from models.color_correction.affine_model import (apply_affine_pil, load_affine_model,
                                                  predict_affine)
import train_color_affine as tca

_SHOW = 480


def _data_url(img, quality=74):
    img = img.convert("RGB").copy()
    img.thumbnail((_SHOW, _SHOW), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_affine_model(CHECKPOINTS_DIR / "color_affine_efficientnet_b0.pt", device)
    assert model is not None, "affine checkpoint missing"

    recs = tca.load_records("val")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(recs)

    sz = 384
    wm = np.ones((sz, sz), dtype=bool)
    if WATERMARK_REGION:
        l, t, r, b = WATERMARK_REGION
        wm[int(t * sz):int(b * sz), int(l * sz):int(r * sz)] = False

    cards = []
    for r in recs:
        if len(cards) >= args.n:
            break
        region = tca._extract_region(r)
        if region is None:
            continue
        dev = develop_raw(r["raw"], size=COLOR_SIZE, neutral=False)
        M = predict_affine(model, dev, device)
        corrected = apply_affine_pil(region, M)
        edited = Image.open(r["edited"]).convert("RGB")

        a = np.asarray(corrected.resize((sz, sz), Image.LANCZOS))
        e = np.asarray(edited.resize((sz, sz), Image.LANCZOS))
        n0 = np.asarray(region.resize((sz, sz), Image.LANCZOS))
        de_model = float(de2000(srgb_to_lab(a)[wm], srgb_to_lab(e)[wm]).mean())
        de_nothing = float(de2000(srgb_to_lab(n0)[wm], srgb_to_lab(e)[wm]).mean())

        cards.append({
            "name": Path(r["raw"]).name,
            "de": de_model, "de0": de_nothing,
            "before": _data_url(region),
            "model": _data_url(corrected),
            "jay": _data_url(edited),
        })
        print(f"  [{len(cards):3d}/{args.n}] {Path(r['raw']).name}  "
              f"dE {de_nothing:.1f} -> {de_model:.1f}", flush=True)

    mean_de = float(np.mean([c["de"] for c in cards]))
    mean_de0 = float(np.mean([c["de0"] for c in cards]))
    cards.sort(key=lambda c: -c["de"])   # worst first for review

    items = ""
    for c in cards:
        color = "#2ecc71" if c["de"] <= 6 else ("#f59e0b" if c["de"] <= 10 else "#e74c3c")
        items += f"""
  <div class="card">
    <div class="row">
      <div class="cell"><div class="tag t0">before</div><img src="{c['before']}" loading="lazy"></div>
      <div class="cell"><div class="tag t1">AI color</div><img src="{c['model']}" loading="lazy"></div>
      <div class="cell"><div class="tag t2">Jay</div><img src="{c['jay']}" loading="lazy"></div>
    </div>
    <div class="meta">
      <span class="fn">{c['name']}</span>
      <span class="de" style="color:{color}">&Delta;E {c['de']:.1f}</span>
      <span class="d0">(before: {c['de0']:.1f})</span>
    </div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Color Preview</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#ddd;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
  #hdr{{background:#0a0a0a;border-bottom:1px solid #222;padding:10px 18px;
        display:flex;align-items:center;gap:18px;position:sticky;top:0;z-index:10}}
  #hdr h1{{font-size:14px;font-weight:600;color:#fff}}
  .stat{{font-size:12px;color:#888}}
  .list{{display:flex;flex-direction:column;gap:10px;padding:12px;max-width:1500px;margin:0 auto}}
  .card{{background:#1a1a1a;border-radius:6px;overflow:hidden;border:1px solid #252525}}
  .row{{display:flex;gap:6px;padding:6px}}
  .cell{{position:relative;flex:1;min-width:0;display:flex;align-items:center;justify-content:center;
        background:#141414;border-radius:4px;overflow:hidden}}
  .cell img{{max-width:100%;max-height:440px;display:block;object-fit:contain}}
  .tag{{position:absolute;top:6px;left:6px;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;z-index:2}}
  .t0{{background:#666;color:#fff}} .t1{{background:#4a90d9;color:#fff}} .t2{{background:#28dc28;color:#052905}}
  .meta{{padding:6px 10px;display:flex;align-items:center;gap:10px;border-top:1px solid #222}}
  .fn{{font-size:11px;color:#666;flex:1}}
  .de{{font-size:13px;font-weight:700}}
  .d0{{font-size:11px;color:#555}}
</style>
</head>
<body>
<div id="hdr">
  <h1>AI Color Preview — {len(cards)} val photos</h1>
  <span class="stat">mean &Delta;E: <b style="color:#fff">{mean_de:.2f}</b> (before: {mean_de0:.2f})</span>
  <span class="stat" style="margin-left:auto">sorted worst-first</span>
</div>
<div class="list">{items}
</div>
</body>
</html>"""
    out = BASE_DIR / "color_preview.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nmean dE: {mean_de:.2f} (before {mean_de0:.2f})  saved: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
