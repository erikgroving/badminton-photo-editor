"""
Color pilot D: predict a global 3x4 color affine per photo, trained in IMAGE
space against Jay's edited JPGs (Lab L1 loss). Bypasses the slider renderer,
whose measured ceiling (GT replay dE2000 = 12.46) barely beats do-nothing
(12.90) while an oracle affine reaches ~5.9.

Input:  full developed frame (camera WB, matches production), 256px.
Output: M[3,4] applied to RGB in linear space: rgb' = M[:, :3] @ rgb + M[:, 3].
Loss:   L1 in CIELAB between affine-applied crop region and Jay's edit,
        watermark region excluded.

Usage:
    python train_color_affine.py                 # train (default 20 epochs)
    python train_color_affine.py --eval          # dE2000 eval of best ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, COLOR_SIZE, WATERMARK_REGION
from data.raw_reader import develop_raw

CKPT = CHECKPOINTS_DIR / "color_affine_efficientnet_b0.pt"
INPUT_SIZE = 256      # full-frame model input
REGION_SIZE = 224     # loss-comparison region size


# ── Torch color space ─────────────────────────────────────────────────────────

def srgb_to_lab_t(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (...,3) in [0,1] -> Lab. Differentiable."""
    c = rgb.clamp(0.0, 1.0)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]],
                     dtype=rgb.dtype, device=rgb.device)
    xyz = lin @ M.T
    wp = torch.tensor([0.95047, 1.0, 1.08883], dtype=rgb.dtype, device=rgb.device)
    t = (xyz / wp).clamp(min=1e-6)
    eps = 216 / 24389
    f = torch.where(t > eps, t ** (1 / 3), (24389 / 27 * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def apply_affine_t(img: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """img: [B,3,H,W] in [0,1] sRGB; M: [B,3,4]. Affine in LINEAR RGB."""
    x = img.clamp(1e-6, 1.0).permute(0, 2, 3, 1) ** 2.2          # [B,H,W,3] linear
    y = torch.einsum("bij,bhwj->bhwi", M[:, :, :3], x) + M[:, None, None, :, 3]
    return (y.clamp(1e-6, 4.0) ** (1 / 2.2)).permute(0, 3, 1, 2).clamp(0.0, 1.0)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_records(split: str) -> list:
    """Axis-aligned AND portrait pairs. Portrait handling: the developed image is
    auto-oriented by rawpy, so the sensor-native crop box is mapped with
    _developed_image_box(flip); for landscape-shot raws that Jay rotated to
    portrait (flip=0, angle~90) the region is rotated CW per the apply_crop
    convention. Every pair is alignment-verified once (luminance cc >= 0.85)
    and the verdict cached, so mis-rotations drop out automatically."""
    color = json.load(open(Path(__file__).parent / "data" / "color_gt.json"))
    crop = {r["raw"]: r for r in json.load(open(Path(__file__).parent / "data" / "crop_gt.json"))}
    recs = []
    for r in color:
        c = crop.get(r["raw"])
        if c is None or r["split"] != split:
            continue
        a = c.get("angle_deg", 0.0)
        if c.get("inlier_ratio", 0.0) < 0.6:
            continue
        if not (abs(a) < 1.0 or abs(a - 90.0) < 1.0):
            continue   # tilted pairs can't be pixel-aligned with axis boxes
        recs.append({"raw": r["raw"], "edited": r["edited"], "box": c["box"],
                     "angle": a})
    return _verified(recs, split)


_PAIR_CACHE = Path(__file__).parent / "cache" / "color_affine_pairs.json"


def _extract_region(r: dict) -> Image.Image | None:
    """Crop region from the oriented develop, rotated to match the edited JPG."""
    from data.raw_reader import get_raw_flip
    from inference.pipeline import _developed_image_box
    dev = develop_raw(r["raw"], size=COLOR_SIZE, neutral=False)
    flip = get_raw_flip(r["raw"])
    box = _developed_image_box(r["box"], flip)
    W, H = dev.size
    px = (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))
    if px[2] - px[0] < 24 or px[3] - px[1] < 24:
        return None
    region = dev.crop(px)
    if flip == 0 and abs(r["angle"] - 90.0) < 1.0:
        region = region.rotate(-90, expand=True)   # apply_crop convention (CW)
    return region


def _verified(recs: list, split: str) -> list:
    """Filter to pairs whose extracted region pixel-aligns with the edited JPG."""
    cache = json.loads(_PAIR_CACHE.read_text()) if _PAIR_CACHE.exists() else {}
    out, dirty = [], False
    for r in recs:
        v = cache.get(r["raw"])
        if v is None:
            region = _extract_region(r)
            if region is None:
                v = False
            else:
                small = region.resize((128, 128), Image.BILINEAR).convert("L")
                edited = Image.open(r["edited"]).convert("L").resize((128, 128), Image.BILINEAR)
                g1 = np.asarray(small, dtype=np.float64).ravel()
                g2 = np.asarray(edited, dtype=np.float64).ravel()
                cc = np.corrcoef(g1, g2)[0, 1]
                v = bool(np.isfinite(cc) and cc >= 0.85)
            cache[r["raw"]] = v
            dirty = True
        if v:
            out.append(r)
    if dirty:
        _PAIR_CACHE.parent.mkdir(exist_ok=True)
        _PAIR_CACHE.write_text(json.dumps(cache))
    print(f"  [{split}] {len(out)}/{len(recs)} pairs alignment-verified")
    return out


class AffineColorDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        dev = develop_raw(r["raw"], size=COLOR_SIZE, neutral=False)  # cached 512
        region = _extract_region(r).resize((REGION_SIZE, REGION_SIZE), Image.LANCZOS)
        edited = Image.open(r["edited"]).convert("RGB").resize(
            (REGION_SIZE, REGION_SIZE), Image.LANCZOS)
        inp = dev.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

        to_t = lambda im: torch.from_numpy(
            np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return to_t(inp), to_t(region), to_t(edited)


def loss_mask(device) -> torch.Tensor:
    """[1,1,H,W] weight mask excluding the watermark region (edited-only)."""
    m = torch.ones(1, 1, REGION_SIZE, REGION_SIZE, device=device)
    if WATERMARK_REGION:
        l, t, r, b = WATERMARK_REGION
        m[:, :, int(t * REGION_SIZE):int(b * REGION_SIZE),
              int(l * REGION_SIZE):int(r * REGION_SIZE)] = 0.0
    return m


# ── Model ─────────────────────────────────────────────────────────────────────

class AffinePredictor(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0"):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=True,
                                          num_classes=0, global_pool="avg")
        self.head = nn.Linear(self.backbone.num_features, 12)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():   # bias -> identity affine
            self.head.bias.copy_(torch.tensor(
                [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]))

    def forward(self, x):
        return self.head(self.backbone(x)).view(-1, 3, 4)


# ── Train / eval ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, mask, opt=None):
    training = opt is not None
    model.train(training)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inp, region, edited in tqdm(loader, leave=False,
                                        desc="train" if training else "val"):
            inp, region, edited = inp.to(device), region.to(device), edited.to(device)
            M = model(inp)
            out = apply_affine_t(region, M)
            lab_o = srgb_to_lab_t(out.permute(0, 2, 3, 1))
            lab_t = srgb_to_lab_t(edited.permute(0, 2, 3, 1))
            per_px = (lab_o - lab_t).abs().mean(-1)              # [B,H,W]
            l = (per_px * mask[0, 0]).sum() / (mask[0, 0].sum() * per_px.shape[0])
            if training:
                opt.zero_grad()
                l.backward()
                opt.step()
            total += l.item() * inp.shape[0]
            n += inp.shape[0]
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--eval", action="store_true",
                    help="dE2000-evaluate best checkpoint against harness baselines")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval:
        evaluate(device)
        return

    tr, va = load_records("train"), load_records("val")
    print(f"train={len(tr)}  val={len(va)}  (axis-aligned well-matched pairs)")
    dl_tr = DataLoader(AffineColorDataset(tr), batch_size=args.batch_size,
                       shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(AffineColorDataset(va), batch_size=args.batch_size,
                       shuffle=False, num_workers=4, pin_memory=True)

    model = AffinePredictor(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mask = loss_mask(device)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_l = run_epoch(model, dl_tr, device, mask, opt)
        va_l = run_epoch(model, dl_va, device, mask)
        sched.step()
        flag = ""
        if va_l < best:
            best = va_l
            torch.save({"model_state": model.state_dict(),
                        "backbone": args.backbone, "epoch": ep,
                        "metrics": {"val_lab_l1": va_l}}, CKPT)
            flag = "  << saved"
        print(f"ep={ep:02d}  train_labL1={tr_l:.3f}  val_labL1={va_l:.3f}{flag}",
              flush=True)
    print(f"\nBest val Lab-L1: {best:.3f}  ckpt: {CKPT.name}")


def evaluate(device):
    """dE2000 on val — portrait-aware, split by crop orientation."""
    from eval_color_imagespace import srgb_to_lab, de2000
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = AffinePredictor(ck["backbone"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    recs = load_records("val")
    rng = np.random.default_rng(42)
    rng.shuffle(recs)
    recs = recs[:160]

    wm_mask = np.ones((REGION_SIZE, REGION_SIZE), dtype=bool)
    if WATERMARK_REGION:
        l, t, r_, b = WATERMARK_REGION
        wm_mask[int(t * REGION_SIZE):int(b * REGION_SIZE),
                int(l * REGION_SIZE):int(r_ * REGION_SIZE)] = False

    rows = []
    for r in recs:
        region = _extract_region(r)
        if region is None:
            continue
        region = region.resize((REGION_SIZE, REGION_SIZE), Image.LANCZOS)
        edited = Image.open(r["edited"]).convert("RGB").resize(
            (REGION_SIZE, REGION_SIZE), Image.LANCZOS)
        dev = develop_raw(r["raw"], size=COLOR_SIZE, neutral=False)
        inp = torch.from_numpy(np.asarray(
            dev.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS),
            dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        reg_t = torch.from_numpy(np.asarray(region, dtype=np.float32) / 255.0
                                 ).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = apply_affine_t(reg_t, model(inp))
        out_u8 = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        lab_t = srgb_to_lab(np.asarray(edited))
        rows.append({
            "portrait": abs(r["angle"] - 90.0) < 1.0,
            "aff": float(de2000(srgb_to_lab(out_u8)[wm_mask], lab_t[wm_mask]).mean()),
            "none": float(de2000(srgb_to_lab(np.asarray(region))[wm_mask],
                                 lab_t[wm_mask]).mean()),
        })

    for label, sel in [("ALL", rows),
                       ("landscape", [x for x in rows if not x["portrait"]]),
                       ("portrait",  [x for x in rows if x["portrait"]])]:
        if not sel:
            continue
        print(f"{label:<10} n={len(sel):<4} do-nothing={np.mean([x['none'] for x in sel]):.2f}"
              f"  affine={np.mean([x['aff'] for x in sel]):.2f}")
    print("(compare: slider GT ceiling 12.46, event-mean 12.45, oracle affine ~5.9)")


if __name__ == "__main__":
    main()
