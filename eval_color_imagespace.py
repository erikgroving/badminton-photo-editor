"""
Image-space evaluation harness for the color-correction stage.

For N val/test pairs:
  develop raw (camera WB, matches training) -> crop to the homography-matched
  region from crop_gt.json -> apply sliders -> compare against Jay's edited JPG
  in CIELAB (D65).

Methods compared:
  nothing     developed raw as-is (do-nothing baseline)
  event_mean  per-event mean GT sliders from the TRAIN split (constant per venue)
  model       current checkpoint (default color_param_efficientnet_b3.pt)
  gt          ground-truth XMP sliders through our renderer (= parameterization+renderer ceiling)
  gt_curve    gt + Jay's tone curve (quantifies how much the missing curve costs)

Metrics per method: dE2000 mean/p95, |dL*|, signed da*/db* (WB cast), chroma delta.
Headline extra: Pearson/Spearman correlation between per-photo param-L1 and
per-photo dE2000 for the model.

Run from anywhere (repo path is hardcoded below):
  python eval_color_imagespace.py --split val --n 120
  python eval_color_imagespace.py --split test --n 0        # 0 = all eligible
  python eval_color_imagespace.py --ckpt checkpoints/color_param_efficientnet_b4.pt
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

from config import COLOR_SIZE, WATERMARK_REGION  # noqa: E402
from data.raw_reader import develop_raw, mask_watermark  # noqa: E402
from data.xmp_reader import PARAM_NAMES, params_to_vec, vec_to_params  # noqa: E402
from inference.apply_params import apply_params_dict  # noqa: E402


# ── Lab / dE2000 (numpy, D65) ─────────────────────────────────────────────────

def srgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """rgb_u8: (...,3) uint8 -> Lab float64."""
    c = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    wp = np.array([0.95047, 1.0, 1.08883])
    t = xyz / wp
    eps, kap = 216 / 24389, 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (kap * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def de2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 between two (...,3) Lab arrays. Returns (...) float."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                   np.where(hsum < 360, hsum / 2 + 180, hsum / 2 - 180))
    hbp = np.where(C1p * C2p == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


# ── Tone curve ────────────────────────────────────────────────────────────────

def apply_tone_curve(img: Image.Image, curve: list | None) -> Image.Image:
    if not curve or len(curve) < 2:
        return img
    xs = np.array([p[0] for p in curve], dtype=np.float64)
    ys = np.array([p[1] for p in curve], dtype=np.float64)
    lut = np.interp(np.arange(256), xs, ys).clip(0, 255).astype(np.uint8)
    arr = np.asarray(img.convert("RGB"))
    return Image.fromarray(lut[arr])


# ── Pair preparation ──────────────────────────────────────────────────────────

def event_of(raw_path: str) -> str:
    """Event id = leading YYYYMM token of the parent folder (venue/lighting)."""
    return Path(raw_path).parent.name.split()[0]


_WB_CACHE_PATH = REPO / "data" / "color_exif_wb_cache.json"
_WB_CACHE = json.load(open(_WB_CACHE_PATH)) if _WB_CACHE_PATH.exists() else {}


def camlog_of(raw_path: str) -> float | None:
    """ln(as-shot R/B gain) from the EXIF WB cache; live rawpy fallback."""
    wb = _WB_CACHE.get(raw_path, {}).get("wb")
    if wb and wb[1] and wb[0] > 0 and wb[2] > 0:
        return float(np.log((wb[0] / wb[1]) / (wb[2] / wb[1])))
    from data.raw_reader import get_as_shot_camlog
    return get_as_shot_camlog(raw_path)


def load_records(split: str) -> tuple[list, dict]:
    color = json.load(open(REPO / "data" / "color_gt.json"))
    crop = {r["raw"]: r for r in json.load(open(REPO / "data" / "crop_gt.json"))}
    recs = []
    for r in color:
        c = crop.get(r["raw"])
        if c is None:
            continue
        r = dict(r)
        r["box"] = c["box"]
        r["angle_deg"] = c.get("angle_deg", 0.0)
        r["inlier_ratio"] = c.get("inlier_ratio", 0.0)
        recs.append(r)
    return [r for r in recs if r["split"] == split], {r["raw"]: r for r in recs}


def eligible(r: dict) -> bool:
    """Landscape, well-matched pairs where crop box coords need no orientation transform."""
    return abs(r["angle_deg"]) < 1.0 and r["inlier_ratio"] >= 0.6


def prepare_pair(r: dict, compare_long: int = 448):
    """Return (raw_region u8, edited_region u8, valid_mask) aligned at same size, or None."""
    dev = develop_raw(r["raw"], size=COLOR_SIZE, neutral=False)   # cached 512-long-edge
    W, H = dev.size
    x1, y1, x2, y2 = r["box"]
    px = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    if px[2] - px[0] < 32 or px[3] - px[1] < 32:
        return None
    region = dev.crop(px)
    w, h = region.size
    scale = compare_long / max(w, h)
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    region = region.resize(size, Image.LANCZOS)
    edited = Image.open(r["edited"]).convert("RGB").resize(size, Image.LANCZOS)

    # Alignment sanity check on luminance
    g1 = np.asarray(region.convert("L"), dtype=np.float64).ravel()
    g2 = np.asarray(edited.convert("L"), dtype=np.float64).ravel()
    cc = np.corrcoef(g1, g2)[0, 1]
    if not np.isfinite(cc) or cc < 0.85:
        return None

    # Exclude watermark region (present in edited only) from all metrics
    mask = np.ones((size[1], size[0]), dtype=bool)
    if WATERMARK_REGION:
        l, t, rr, bb = WATERMARK_REGION
        mask[int(t * size[1]):int(bb * size[1]), int(l * size[0]):int(rr * size[0])] = False
    return region, edited, mask


def model_input(r: dict, bboxes: dict) -> Image.Image:
    """Replicate ColorDataset training input: dev(camera-WB) + watermark mask + player crop."""
    img = develop_raw(r["raw"], size=COLOR_SIZE, neutral=False)
    img = mask_watermark(img)
    bbox = bboxes.get(r["raw"])
    if bbox:
        W, H = img.size
        x1, y1, x2, y2 = bbox
        pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        img = img.crop((int(max(0, x1 - pw) * W), int(max(0, y1 - ph) * H),
                        int(min(1, x2 + pw) * W), int(min(1, y2 + ph) * H)))
    return img.resize(COLOR_SIZE, Image.LANCZOS)


# ── Metrics ───────────────────────────────────────────────────────────────────

def image_metrics(pred: Image.Image, target: Image.Image, mask: np.ndarray) -> dict:
    lab_p = srgb_to_lab(np.asarray(pred))[mask]
    lab_t = srgb_to_lab(np.asarray(target))[mask]
    de = de2000(lab_p, lab_t)
    dl = lab_p[:, 0] - lab_t[:, 0]
    da = lab_p[:, 1] - lab_t[:, 1]
    db = lab_p[:, 2] - lab_t[:, 2]
    dc = np.hypot(lab_p[:, 1], lab_p[:, 2]) - np.hypot(lab_t[:, 1], lab_t[:, 2])
    return {
        "de2000_mean": float(de.mean()),
        "de2000_p95": float(np.percentile(de, 95)),
        "abs_dL": float(np.abs(dl).mean()),
        "dL": float(dl.mean()),
        "da": float(da.mean()),          # + = pred redder than Jay
        "db": float(db.mean()),          # + = pred yellower than Jay
        "wb_ab_mag": float(np.hypot(da.mean(), db.mean())),
        "dchroma": float(dc.mean()),     # + = pred more saturated than Jay
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=120, help="0 = all eligible")
    ap.add_argument("--ckpt", default=str(REPO / "checkpoints" / "color_param_efficientnet_b3.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    import torch
    from torchvision import transforms
    from models.color_correction.param_model import build_param_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_param_model(pretrained=False, backbone_name=ck.get("backbone"))
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    print(f"Checkpoint: {Path(args.ckpt).name}  backbone={ck.get('backbone')} "
          f"val_l1={ck.get('metrics', {}).get('val_l1', float('nan')):.4f}")

    recs, _ = load_records(args.split)
    bboxes = json.load(open(REPO / "data" / "player_bboxes.json")) \
        if (REPO / "data" / "player_bboxes.json").exists() else {}

    # Per-event mean GT sliders computed from the TRAIN split
    train_recs, _ = load_records("train")
    ev_params: dict[str, dict] = {}
    ev_group = defaultdict(list)
    for r in train_recs:
        ev_group[event_of(r["raw"])].append(r["params"])
    for ev, plist in ev_group.items():
        ev_params[ev] = {k: float(np.mean([p[k] for p in plist])) for k in PARAM_NAMES}

    elig = [r for r in recs if eligible(r)]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(elig)
    if args.n > 0:
        elig = elig[: args.n * 2]     # oversample; alignment check drops some

    tf = transforms.ToTensor()
    per_photo = []
    n_target = args.n if args.n > 0 else len(elig)
    skipped = 0

    for r in elig:
        if len(per_photo) >= n_target:
            break
        pair = prepare_pair(r)
        if pair is None:
            skipped += 1
            continue
        region, edited, mask = pair

        with torch.no_grad():
            inp = tf(model_input(r, bboxes)).unsqueeze(0).to(device)
            pred_vec = model(inp).squeeze(0).cpu().tolist()
        gt_vec = params_to_vec(r["params"])
        param_l1 = float(np.mean(np.abs(np.array(pred_vec) - np.array(gt_vec))))
        per_param_err = np.abs(np.array(pred_vec) - np.array(gt_vec))

        ev = event_of(r["raw"])
        # camlog enables the calibrated differential Temperature application;
        # camlog=None skips the temperature shift (the "no-WB" diagnostic).
        camlog = camlog_of(r["raw"])
        outputs = {
            "nothing":    region,
            "event_mean": apply_params_dict(region, ev_params.get(ev, {}), camlog=camlog),
            "model":      apply_params_dict(region, vec_to_params(pred_vec), camlog=camlog),
            "gt":         apply_params_dict(region, r["params"], camlog=camlog),
            "gt_no_wb":   apply_params_dict(region, r["params"], camlog=None),
            "gt_curve":   apply_tone_curve(apply_params_dict(region, r["params"], camlog=camlog),
                                           r.get("tone_curve")),
        }
        row = {"raw": r["raw"], "event": ev, "param_l1": param_l1,
               "per_param_err": per_param_err.tolist()}
        for name, img in outputs.items():
            row[name] = image_metrics(img, edited, mask)
        per_photo.append(row)
        if len(per_photo) % 20 == 0:
            print(f"  {len(per_photo)}/{n_target} pairs done")

    print(f"\nEvaluated {len(per_photo)} pairs ({skipped} skipped: misaligned/degenerate)")

    # ── Aggregate ────────────────────────────────────────────────────────────
    methods = ["nothing", "event_mean", "model", "gt", "gt_no_wb", "gt_curve"]
    agg = {}
    for m in methods:
        agg[m] = {k: float(np.mean([p[m][k] for p in per_photo]))
                  for k in per_photo[0][m]}
        agg[m]["de2000_p95_mean"] = agg[m].pop("de2000_p95")

    print(f"\n{'method':<12}{'dE2000':>8}{'dE p95':>8}{'|dL|':>7}{'da':>7}{'db':>7}{'dChroma':>9}")
    for m in methods:
        a = agg[m]
        print(f"{m:<12}{a['de2000_mean']:>8.2f}{a['de2000_p95_mean']:>8.2f}"
              f"{a['abs_dL']:>7.2f}{a['da']:>7.2f}{a['db']:>7.2f}{a['dchroma']:>9.2f}")

    # Per-event breakdown
    print(f"\nPer-event dE2000 mean:")
    events = sorted({p["event"] for p in per_photo})
    print(f"{'event':<10}" + "".join(f"{m:>12}" for m in methods) + f"{'n':>5}")
    per_event = {}
    for ev in events:
        rows = [p for p in per_photo if p["event"] == ev]
        vals = {m: float(np.mean([p[m]["de2000_mean"] for p in rows])) for m in methods}
        per_event[ev] = {**vals, "n": len(rows)}
        print(f"{ev:<10}" + "".join(f"{vals[m]:>12.2f}" for m in methods) + f"{len(rows):>5}")

    # Correlation: param-L1 vs image dE2000 (model)
    l1s = np.array([p["param_l1"] for p in per_photo])
    des = np.array([p["model"]["de2000_mean"] for p in per_photo])
    pear = float(np.corrcoef(l1s, des)[0, 1])
    spear = spearman(l1s, des)
    print(f"\nparam-L1 vs model dE2000:  Pearson r={pear:.3f}  Spearman rho={spear:.3f}")
    print(f"param-L1 mean={l1s.mean():.4f}  model dE2000 mean={des.mean():.2f}")

    # Per-slider normalized error
    pp = np.mean(np.array([p["per_param_err"] for p in per_photo]), axis=0)
    print("\nPer-slider mean |pred-gt| (normalized units):")
    for name, e in sorted(zip(PARAM_NAMES, pp), key=lambda t: -t[1]):
        print(f"  {name:<22}{e:.4f}")

    out_path = Path(args.out) if args.out else Path(__file__).parent / \
        f"color_eval_{args.split}_{Path(args.ckpt).stem}.json"
    json.dump({"checkpoint": args.ckpt, "split": args.split,
               "n": len(per_photo), "skipped": skipped,
               "aggregate": agg, "per_event": per_event,
               "correlation": {"pearson": pear, "spearman": spear},
               "per_slider_l1": dict(zip(PARAM_NAMES, pp.tolist())),
               "per_photo": per_photo},
              open(out_path, "w"), indent=1)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
