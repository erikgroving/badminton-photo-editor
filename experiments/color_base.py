"""
Shared infrastructure for color-correction conditioning experiments (c1–c6).

Each experiment defines a feature_fn(raw_path, thumb_pil, cache) -> list[float]
and calls train_conditioned() with that function.
"""
import io
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, COLOR_GT_FILE, COLOR_SIZE
from data.raw_reader import develop_raw, mask_watermark
from data.split_lookup import build_split_lookup
from data.xmp_reader import params_to_vec

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

BACKBONE    = "efficientnet_b3"
EPOCHS      = 30
BATCH_SIZE  = 16
LR          = 1e-4
COND_EMB    = 64          # conditioning embedding dim
NUM_PARAMS  = 14          # len(COLOR_PARAM_NAMES)
LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ── Feature cache builder ──────────────────────────────────────────────────────

def build_exif_wb_cache(records: list[dict]) -> dict:
    """
    Pre-cache EXIF + rawpy WB for all records.
    Returns dict keyed by raw_path with exif/wb data.
    """
    cache_path = ROOT / "data" / "color_exif_wb_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import rawpy
    cache = {}
    print("  Building EXIF/WB cache …", flush=True)
    for r in tqdm(records, desc="EXIF/WB cache"):
        raw_p = r["raw"]
        entry: dict = {}
        try:
            with rawpy.imread(raw_p) as raw:
                entry["wb"]  = list(raw.camera_whitebalance)
                entry["dwb"] = list(raw.daylight_whitebalance)
                thumb_bytes  = raw.extract_thumb().data
            thumb_img = Image.open(io.BytesIO(thumb_bytes))
            exif = thumb_img._getexif() or {}
            entry["iso"]     = float(exif.get(34855, 800) or 800)
            entry["shutter"] = float(exif.get(33434, 0.001) or 0.001)
            entry["fnum"]    = float(exif.get(33437, 2.8) or 2.8)
        except Exception:
            entry = {"wb": [2048, 1024, 3000, 1024], "dwb": [2.0, 1.0, 1.5, 0.0],
                     "iso": 800.0, "shutter": 0.001, "fnum": 2.8}
        cache[raw_p] = entry

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"  Cached {len(cache)} entries → {cache_path}")
    return cache


# ── Feature extractors ─────────────────────────────────────────────────────────

def features_exif(raw_path: str, thumb: Image.Image, cache: dict) -> list:
    """c1: ISO, shutter speed, aperture → 3 dim."""
    e = cache.get(raw_path, {})
    iso     = float(e.get("iso", 800))
    shutter = float(e.get("shutter", 0.001))
    fnum    = float(e.get("fnum", 2.8))
    iso_n   = min(1.0, math.log2(max(iso, 100) / 100) / 9.0)   # ISO100→0, 51200→1
    shut_n  = min(1.0, -math.log2(max(shutter, 1/8000)) / 13.0) # 1s→0, 1/8000→1
    f_n     = min(1.0, (fnum - 1.0) / 15.0)
    return [iso_n, shut_n, f_n]


def features_camera_wb(raw_path: str, thumb: Image.Image, cache: dict) -> list:
    """c2: Camera WB + daylight WB ratios → 4 dim."""
    e  = cache.get(raw_path, {})
    wb = e.get("wb",  [2048.0, 1024.0, 3000.0, 1024.0])
    r, g1, b, g2 = [float(x) for x in wb]
    g = (g1 + g2) / 2 if g2 > 0 else g1
    rg = r / max(g, 1)
    bg = b / max(g, 1)
    dwb = e.get("dwb", [2.0, 1.0, 1.5, 0.0])
    dr, dg1, db, _ = [float(x) for x in dwb]
    dg = dg1 if dg1 > 0 else 1.0
    drg = dr / max(dg, 0.01)
    dbg = db / max(dg, 0.01)
    # Normalise roughly to [0,1]: rg ∈ [0.5,3], bg ∈ [0.5,4]
    return [
        (rg - 0.5) / 2.5,
        (bg - 0.5) / 3.5,
        (drg - 0.5) / 2.5,
        (dbg - 0.5) / 3.5,
    ]


def features_classic_wb(raw_path: str, thumb: Image.Image, cache: dict) -> list:
    """c3: Gray-world, max-RGB, p95 WB estimates from thumbnail → 6 dim."""
    arr = np.array(thumb.resize((128, 128), Image.LANCZOS), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    def _ratio(a, b_ch):
        return float(np.mean(a)) / max(float(np.mean(b_ch)), 1e-5)

    def _ratio_max(a, b_ch):
        return float(np.max(a)) / max(float(np.max(b_ch)), 1e-5)

    def _ratio_p95(a, b_ch):
        return float(np.percentile(a, 95)) / max(float(np.percentile(b_ch, 95)), 1e-5)

    gw_rg = _ratio(r, g);    gw_bg = _ratio(b, g)
    mx_rg = _ratio_max(r, g); mx_bg = _ratio_max(b, g)
    p5_rg = _ratio_p95(r, g); p5_bg = _ratio_p95(b, g)
    # Each ratio is near 1.0 for neutral; clip to [0.3, 3.0] → normalise to [0,1]
    def _norm(v):
        return float(np.clip((v - 0.3) / 2.7, 0.0, 1.0))
    return [_norm(gw_rg), _norm(gw_bg), _norm(mx_rg), _norm(mx_bg),
            _norm(p5_rg), _norm(p5_bg)]


def features_region_stats(raw_path: str, thumb: Image.Image, cache: dict) -> list:
    """c4: 3×3 grid of (mean brightness, mean saturation) → 18 dim."""
    small = thumb.resize((96, 96), Image.LANCZOS).convert("HSV")
    arr   = np.array(small, dtype=np.float32) / 255.0
    h_ch, s_ch, v_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    feats = []
    for row in range(3):
        for col in range(3):
            rs, re = row * 32, (row + 1) * 32
            cs, ce = col * 32, (col + 1) * 32
            feats.append(float(v_ch[rs:re, cs:ce].mean()))
            feats.append(float(s_ch[rs:re, cs:ce].mean()))
    return feats  # 18 dim


def features_histogram(raw_path: str, thumb: Image.Image, cache: dict) -> list:
    """c5: 16-bin normalised histograms for R, G, B → 48 dim."""
    arr  = np.array(thumb.resize((128, 128), Image.LANCZOS), dtype=np.float32)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        feats.extend(hist.tolist())
    return feats  # 48 dim


def features_combined(raw_path: str, thumb: Image.Image, cache: dict) -> list:
    """c6: c1 + c2 + c3 + c4 + c5 → 3+4+6+18+48 = 79 dim."""
    return (features_exif(raw_path, thumb, cache)
            + features_camera_wb(raw_path, thumb, cache)
            + features_classic_wb(raw_path, thumb, cache)
            + features_region_stats(raw_path, thumb, cache)
            + features_histogram(raw_path, thumb, cache))


# ── Dataset ────────────────────────────────────────────────────────────────────

_PLAYER_BBOXES: dict = {}

def _load_player_bboxes() -> dict:
    p = ROOT / "data" / "player_bboxes.json"
    return json.load(open(p)) if p.exists() else {}

def _crop_to_player(img: Image.Image, bbox: list, pad: float = 0.15) -> Image.Image:
    W, H = img.size
    x1n, y1n, x2n, y2n = bbox
    px = (x2n - x1n) * pad; py = (y2n - y1n) * pad
    x1 = max(0.0, x1n - px); y1 = max(0.0, y1n - py)
    x2 = min(1.0, x2n + px); y2 = min(1.0, y2n + py)
    return img.crop((int(x1*W), int(y1*H), int(x2*W), int(y2*H)))


class ColorCondDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], feature_fn: Callable,
                 exif_wb_cache: dict, augment: bool = False,
                 input_size: tuple = COLOR_SIZE):
        global _PLAYER_BBOXES
        self.records      = records
        self.feature_fn   = feature_fn
        self.cache        = exif_wb_cache
        self.augment      = augment
        self.input_size   = input_size
        if not _PLAYER_BBOXES:
            _PLAYER_BBOXES = _load_player_bboxes()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        raw_pil    = develop_raw(r["raw"], size=self.input_size, neutral=False)
        edited_pil = Image.open(r["edited"]).convert("RGB").resize(self.input_size, Image.LANCZOS)
        raw_pil    = mask_watermark(raw_pil)
        edited_pil = mask_watermark(edited_pil)

        bbox = _PLAYER_BBOXES.get(r["raw"])
        if bbox:
            thumb = _crop_to_player(raw_pil, bbox)
            raw_pil    = thumb.resize(self.input_size, Image.LANCZOS)
            edited_pil = _crop_to_player(edited_pil, bbox).resize(self.input_size, Image.LANCZOS)
        else:
            thumb = raw_pil

        if self.augment and torch.rand(1).item() > 0.5:
            raw_pil    = raw_pil.transpose(Image.FLIP_LEFT_RIGHT)
            edited_pil = edited_pil.transpose(Image.FLIP_LEFT_RIGHT)

        cond = torch.tensor(self.feature_fn(r["raw"], thumb, self.cache),
                            dtype=torch.float32)
        raw_t    = self.tf(raw_pil)
        param_vec = torch.tensor(params_to_vec(r["params"]), dtype=torch.float32)
        return raw_t, cond, param_vec


# ── Model ──────────────────────────────────────────────────────────────────────

class ConditionedParamModel(nn.Module):
    def __init__(self, backbone_name: str, cond_dim: int,
                 cond_emb: int = COND_EMB, num_params: int = NUM_PARAMS):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True,
                                          num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_emb * 2), nn.ReLU(),
            nn.Linear(cond_emb * 2, cond_emb), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + cond_emb, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_params), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        feat     = self.backbone(x)
        cond_emb = self.cond_encoder(cond)
        return self.head(torch.cat([feat, cond_emb], dim=1))


# ── Training loop ──────────────────────────────────────────────────────────────

def train_conditioned(
    exp_name: str,
    feature_fn: Callable,
    cond_dim: int,
    backbone: str = BACKBONE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> dict:
    from config import COLOR_GT_FILE
    with open(COLOR_GT_FILE) as f:
        records = json.load(f)
    split_lookup = build_split_lookup()
    train_recs = [r for r in records if split_lookup.get(r["raw"]) == "train"]
    val_recs   = [r for r in records if split_lookup.get(r["raw"]) == "val"]
    test_recs  = [r for r in records if split_lookup.get(r["raw"]) == "test"]
    print(f"  [{exp_name}] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    cache = build_exif_wb_cache(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    model = ConditionedParamModel(backbone, cond_dim).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = DataLoader(
        ColorCondDataset(train_recs, feature_fn, cache, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        ColorCondDataset(val_recs, feature_fn, cache),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        ColorCondDataset(test_recs, feature_fn, cache),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ckpt_path = CHECKPOINTS_DIR / f"color_param_{backbone}_{exp_name}.pt"
    best_l1   = float("inf")
    best_meta: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        sum_l1 = 0.0
        for raw_t, cond_t, param_vec in tqdm(train_loader,
                                              desc=f"[{exp_name}] ep{epoch}/{epochs}",
                                              leave=False):
            raw_t, cond_t, param_vec = raw_t.to(device), cond_t.to(device), param_vec.to(device)
            opt.zero_grad()
            loss = F.l1_loss(model(raw_t, cond_t), param_vec)
            loss.backward()
            opt.step()
            sum_l1 += loss.item()
        sched.step()

        model.eval()
        v_l1 = 0.0
        with torch.no_grad():
            for raw_t, cond_t, param_vec in val_loader:
                v_l1 += F.l1_loss(model(raw_t.to(device), cond_t.to(device)),
                                   param_vec.to(device)).item()
        v_l1 /= len(val_loader)
        print(f"  [{exp_name}] ep{epoch:02d}  train_l1={sum_l1/len(train_loader):.4f}"
              f"  val_l1={v_l1:.4f}", flush=True)

        if v_l1 < best_l1:
            best_l1  = v_l1
            best_meta = {"exp": exp_name, "backbone": backbone, "epoch": epoch,
                         "cond_dim": cond_dim,
                         "metrics": {"val_l1": v_l1, "n": len(val_recs)}}
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "backbone": backbone, "exp": exp_name,
                        "cond_dim": cond_dim, "metrics": best_meta["metrics"]},
                       ckpt_path)
            print(f"    [OK] best val_l1={best_l1:.4f}  saved → {ckpt_path.name}")

    # Test eval
    t_l1 = 0.0
    with torch.no_grad():
        for raw_t, cond_t, param_vec in test_loader:
            t_l1 += F.l1_loss(model(raw_t.to(device), cond_t.to(device)),
                               param_vec.to(device)).item()
    t_l1 /= len(test_loader)
    best_meta["test_l1"] = t_l1
    print(f"  [{exp_name}] DONE  best_val={best_l1:.4f}  test={t_l1:.4f}")
    return best_meta
