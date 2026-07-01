"""
Experiment 5 — Image region statistics as auxiliary conditioning.

Hypothesis: Jay's cropping is influenced by background content — crowd density,
background complexity, court visibility. Encoding per-region image statistics
gives the model this context without needing object detection.

Architecture:
  efficientnet_b3 backbone (pretrained, global avg pool)
  + player_encoder:  Linear(4  -> 32) + ReLU   [player bbox conditioning]
  + stats_encoder:   Linear(27 -> 32) + ReLU   [NEW: per-region image stats]
  box_head input: backbone_feats(1536) + player_emb(32) + stats_emb(32)
  box_head: Linear->256->ReLU->Dropout(0.3)->Linear->4->Sigmoid

Stats features (27-dim, computed on-the-fly from raw PIL thumbnail):
  9 regions (3x3 grid) × 3 statistics each:
    - Mean brightness (Y channel from YCbCr)
    - Edge density    (Sobel gradient magnitude mean)
    - Color saturation mean (S from HSV)
  All values in [0, 1] — scale-invariant, no normalisation beyond clamping.

Training:
  backbone:    efficientnet_b3
  epochs:      10
  batch_size:  16
  lr:          1e-4
  loss:        CropLoss (alpha=0.5, player_weight=0.5) — no angle head
  log:         logs/exp5_imgstats.log
  ckpt_tag:    _exp5_imgstats
  ckpt:        checkpoints/cropping_efficientnet_b3_exp5_imgstats.pt

Baseline: efficientnet_b3, 25ep, test IoU=0.819

Usage:
    python experiments/exp5_image_stats.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import CropLoss, box_iou_numpy

# ── Constants ─────────────────────────────────────────────────────────────────
BACKBONE       = "efficientnet_b3"
EPOCHS         = 10
BATCH_SIZE     = 16
LR             = 1e-4
CKPT_TAG       = "_exp5_imgstats"
NUM_STATS      = 27          # 9 regions × 3 stats (brightness, edge, saturation)
PLAYER_EMB_DIM = 32
STATS_EMB_DIM  = 32
EXTRACT_SIZE   = 512

_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_LOG_FILE           = ROOT / "logs" / "exp5_imgstats.log"
_CKPT_PATH          = CHECKPOINTS_DIR / f"cropping_{BACKBONE}{CKPT_TAG}.pt"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(str(_LOG_FILE), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Image statistics computation ──────────────────────────────────────────────

def compute_region_stats(img: Image.Image, grid: int = 3) -> list:
    """
    Divide `img` into a grid×grid array of regions. For each region compute:
      - Mean brightness  (Y channel from YCbCr, range [0,1])
      - Edge density     (mean Sobel gradient magnitude, range [0,1])
      - Color saturation (S from HSV, range [0,1])

    Returns a flat list of grid*grid*3 floats, all in [0, 1].
    Computed BEFORE the torchvision transform — uses the raw PIL thumbnail.
    """
    w, h = img.size
    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # Grayscale for brightness and edges
    gray = img_np.mean(axis=2)  # (H, W)

    # Edge density: simple absolute gradient approximation (prepend to stay same shape)
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edges = (gx + gy)  # range [0, ~2]; empirically stays well below 1 for photos

    # Saturation from HSV: S = (max - min) / max
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-6), 0.0)

    feats = []
    rh = h // grid
    rw = w // grid
    for i in range(grid):
        for j in range(grid):
            r_gray = gray [i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
            r_edge = edges[i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
            r_sat  = sat  [i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
            feats.extend([
                float(r_gray.mean()),
                float(np.clip(r_edge.mean(), 0.0, 1.0)),
                float(r_sat.mean()),
            ])
    return feats  # 27 values


# ── Model ─────────────────────────────────────────────────────────────────────

def build_stats_model(backbone: str = BACKBONE, pretrained: bool = True) -> nn.Module:
    """
    CropRegressor conditioned on player bbox + per-region image statistics.
    Forward signature: model(x, player_bbox, stats)
    """
    try:
        backbone_model = timm.create_model(backbone, pretrained=pretrained,
                                           num_classes=0, global_pool="avg")
    except RuntimeError:
        backbone_model = timm.create_model(backbone, pretrained=False,
                                           num_classes=0, global_pool="avg")

    in_features = backbone_model.num_features
    head_in     = in_features + PLAYER_EMB_DIM + STATS_EMB_DIM

    class StatsCondCropRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone       = backbone_model
            self.player_encoder = nn.Sequential(
                nn.Linear(4, PLAYER_EMB_DIM),
                nn.ReLU(),
            )
            self.stats_encoder = nn.Sequential(
                nn.Linear(NUM_STATS, STATS_EMB_DIM),
                nn.ReLU(),
            )
            self.box_head = nn.Sequential(
                nn.Linear(head_in, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor,
                    player_bbox: torch.Tensor,
                    stats: torch.Tensor) -> torch.Tensor:
            feats      = self.backbone(x)
            player_emb = self.player_encoder(player_bbox)
            stats_emb  = self.stats_encoder(stats)
            combined   = torch.cat([feats, player_emb, stats_emb], dim=1)
            return self.box_head(combined)

        def set_grad_checkpointing(self, enable: bool = True) -> None:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=enable)

    return StatsCondCropRegressor()


# ── Dataset ───────────────────────────────────────────────────────────────────

# Fallback stats for images where thumbnail extraction fails (mid-grey, low edge, mid-sat)
_STATS_FALLBACK = [0.5, 0.05, 0.4] * 9  # 27 values

class StatsCondCropDataset(Dataset):
    def __init__(
        self,
        records: list,
        transform,
        hflip: bool = False,
        player_bbox_cache: dict | None = None,
    ):
        self.records           = records
        self.transform         = transform
        self.hflip             = hflip
        self.player_bbox_cache = player_bbox_cache or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)
        box = list(r["box"])  # [x1, y1, x2, y2] in [0,1]

        # Compute image stats from the RAW PIL thumbnail (before torchvision transform)
        try:
            stats = compute_region_stats(img, grid=3)
        except Exception:
            stats = list(_STATS_FALLBACK)

        # Player bbox for model conditioning and clipping penalty
        pb = self.player_bbox_cache.get(r["raw"])
        player_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]

        if self.hflip and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x1, y1, x2, y2 = box
            box = [1.0 - x2, y1, 1.0 - x1, y2]
            if pb is not None:
                px1, py1, px2, py2 = player_bbox
                player_bbox = [1.0 - px2, py1, 1.0 - px1, py2]
            # Flip the 3x3 grid left-right: for each row, swap col0 <-> col2
            # Stats layout: [brightness_r0c0, edge_r0c0, sat_r0c0,  brightness_r0c1, ..., sat_r2c2]
            # i.e., outer loop = row (i), inner loop = col (j), innermost = stat (k)
            # Index: (i*3 + j)*3 + k
            for i in range(3):
                for k in range(3):
                    left_idx  = (i * 3 + 0) * 3 + k
                    right_idx = (i * 3 + 2) * 3 + k
                    stats[left_idx], stats[right_idx] = stats[right_idx], stats[left_idx]

        img_t        = self.transform(img)
        box_t        = torch.tensor(box,         dtype=torch.float32)
        player_bbox_t = torch.tensor(player_bbox, dtype=torch.float32)
        stats_t      = torch.tensor(stats,       dtype=torch.float32)
        return img_t, box_t, player_bbox_t, stats_t


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for imgs, boxes, player_bboxes, stats in loader:
            box_pred = model(imgs.to(device),
                             player_bboxes.to(device),
                             stats.to(device))
            all_pred.append(box_pred.cpu().numpy())
            all_gt.append(boxes.numpy())
    pred_arr = np.concatenate(all_pred)
    gt_arr   = np.concatenate(all_gt)
    ious     = box_iou_numpy(pred_arr, gt_arr)
    return {
        "mean_iou":   float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou_gt70":   float((ious >= 0.70).mean()),
        "iou_gt80":   float((ious >= 0.80).mean()),
        "n":          len(ious),
    }


# ── Feature correlation analysis ──────────────────────────────────────────────

def _analyze_feature_correlation(records: list, player_bbox_cache: dict) -> None:
    """
    Compute 27 image stats for each training record, then correlate each stat
    with the GT crop box coordinates. Reports:
      - Top correlating stats (most informative)
      - Per-region variance (which regions are most discriminative)
    """
    COORD_NAMES = ["x1", "y1", "x2", "y2"]
    STAT_NAMES  = []
    for i in range(3):
        for j in range(3):
            STAT_NAMES += [f"bright_r{i}c{j}", f"edge_r{i}c{j}", f"sat_r{i}c{j}"]

    feat_rows, box_rows = [], []
    log.info("  [correlation] Computing stats for training records (this takes a moment)...")
    n_fail = 0
    for r in records:
        try:
            img = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)
            st  = compute_region_stats(img, grid=3)
            feat_rows.append(st)
            box_rows.append(r["box"])
        except Exception:
            n_fail += 1

    if not feat_rows:
        log.info("  [correlation] No samples computed — skipping.")
        return

    if n_fail:
        log.info(f"  [correlation] {n_fail} records failed thumbnail extraction (skipped).")

    F = np.array(feat_rows, dtype=np.float32)  # (N, 27)
    B = np.array(box_rows,  dtype=np.float32)  # (N, 4)

    log.info(f"\n  [Feature-Crop Correlation Analysis]  N={len(feat_rows)}")
    log.info(f"  Stats shape: {F.shape}  |  Box shape: {B.shape}")

    # Per-region variance (averaged across 3 stats per region)
    log.info("\n  Per-region mean variance (higher = more discriminative):")
    region_vars = []
    for i in range(3):
        for j in range(3):
            base = (i * 3 + j) * 3
            region_feat_var = F[:, base:base + 3].var(axis=0).mean()
            region_vars.append((region_feat_var, f"r{i}c{j}"))
    region_vars.sort(key=lambda x: x[0], reverse=True)
    for var, name in region_vars:
        log.info(f"    region {name}: mean_var={var:.5f}")

    # Top correlations with crop coordinates
    top: list[tuple[float, str, str]] = []
    for bi, coord in enumerate(COORD_NAMES):
        b_col = B[:, bi] - B[:, bi].mean()
        b_std = b_col.std() + 1e-8
        for fi, fname in enumerate(STAT_NAMES):
            f_col = F[:, fi] - F[:, fi].mean()
            f_std = f_col.std() + 1e-8
            corr  = float(np.mean(f_col * b_col) / (f_std * b_std))
            top.append((abs(corr), corr, fname, coord))

    top.sort(key=lambda x: x[0], reverse=True)
    log.info("\n  Top 20 feature-crop correlations (|r|):")
    for abs_r, r_val, fname, coord in top[:20]:
        sign = "+" if r_val >= 0 else "-"
        log.info(f"    |r|={abs_r:.3f}  ({sign})  {fname} -> {coord}")

    # Which stat type dominates: brightness vs edge vs sat
    by_type = {"bright": [], "edge": [], "sat": []}
    for abs_r, _, fname, _ in top:
        if fname.startswith("bright"):
            by_type["bright"].append(abs_r)
        elif fname.startswith("edge"):
            by_type["edge"].append(abs_r)
        else:
            by_type["sat"].append(abs_r)
    log.info("\n  Mean |r| by stat type:")
    for stype, vals in by_type.items():
        log.info(f"    {stype:8s}: mean_|r|={np.mean(vals):.4f}  max_|r|={np.max(vals):.4f}")


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Experiment 5 (imgstats): 27-dim per-region image stat conditioning")
    log.info(f"  backbone={BACKBONE}  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    log.info(f"  ckpt_tag={CKPT_TAG}  num_stats={NUM_STATS}")
    log.info("=" * 70)

    # Load GT
    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    log.info(f"  train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    # Load primary player bbox cache
    player_bbox_cache: dict = {}
    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        n_cov = sum(1 for r in all_records if player_bbox_cache.get(r["raw"]) is not None)
        log.info(f"  Primary bbox cache: {len(player_bbox_cache):,} entries "
                 f"({n_cov}/{len(all_records)} covered)")
    else:
        log.info(f"  Warning: primary bbox cache not found at {_PRIMARY_BBOX_CACHE}")

    # Feature correlation analysis on a sample of training records (first 200 for speed)
    sample = train_recs[:200]
    log.info(f"\n  Running correlation analysis on {len(sample)} training samples...")
    _analyze_feature_correlation(sample, player_bbox_cache)

    # Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n  device={device}")
    model = build_stats_model(backbone=BACKBONE, pretrained=True)
    model = model.to(device)

    # Input transforms from timm model config
    data_cfg   = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean  = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    norm_std   = tuple(data_cfg.get("std",  _IMAGENET_STD))
    log.info(f"  input_size={input_size}  norm_mean={norm_mean}  norm_std={norm_std}")
    log.info(f"  model params: {sum(p.numel() for p in model.parameters()):,}")

    tf_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(list(norm_mean), list(norm_std)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(list(norm_mean), list(norm_std)),
    ])

    # num_workers=0 required on Windows: avoids CUDA illegal-memory-access in forked workers
    nw = 0
    train_loader = DataLoader(
        StatsCondCropDataset(train_recs, tf_train, hflip=True,
                             player_bbox_cache=player_bbox_cache),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=False,
    )
    val_loader = DataLoader(
        StatsCondCropDataset(val_recs, tf_val,
                             player_bbox_cache=player_bbox_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=False,
    )
    test_loader = DataLoader(
        StatsCondCropDataset(test_recs, tf_val,
                             player_bbox_cache=player_bbox_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=False,
    )

    criterion = CropLoss(alpha=0.5, angle_weight=0.25, player_weight=0.5)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_iou     = -1.0
    best_metrics: dict = {}

    log.info(f"\n  Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, boxes, player_bboxes, stats in tqdm(
                train_loader, desc=f"[exp5] ep{epoch}/{EPOCHS}", leave=False):
            imgs          = imgs.to(device)
            boxes         = boxes.to(device)
            player_bboxes = player_bboxes.to(device)
            stats_t       = stats.to(device)
            optimizer.zero_grad()
            box_pred = model(imgs, player_bboxes, stats_t)
            loss     = criterion(box_pred, boxes, player_bbox=player_bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_m = _evaluate(model, val_loader, device)
        log.info(
            f"  ep{epoch:02d}"
            f"  loss={total_loss / len(train_loader):.4f}"
            f"  [val] mean_iou={val_m['mean_iou']:.4f}"
            f"  med={val_m['median_iou']:.4f}"
            f"  >0.7:{val_m['iou_gt70']:.1%}"
            f"  >0.8:{val_m['iou_gt80']:.1%}"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m, backbone=BACKBONE, epoch=epoch)
            torch.save({
                "epoch":        epoch,
                "backbone":     BACKBONE,
                "ckpt_tag":     CKPT_TAG,
                "model_state":  model.state_dict(),
                "metrics":      val_m,
                "input_size":   input_size,
                "norm_mean":    norm_mean,
                "norm_std":     norm_std,
                "num_stats":    NUM_STATS,
            }, str(_CKPT_PATH))
            log.info(f"    [OK] Saved best (mean_iou={best_iou:.4f})")

    # Final test evaluation using best checkpoint
    ck = torch.load(str(_CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = _evaluate(model, test_loader, device)
    best_metrics["test_metrics"] = test_m

    BASELINE_TEST_IOU = 0.819

    log.info("\n" + "=" * 70)
    log.info("[exp5 imgstats] Results summary")
    log.info(f"  BEST VAL   mean_iou={best_metrics['mean_iou']:.4f}"
             f"  median={best_metrics['median_iou']:.4f}"
             f"  >0.7:{best_metrics['iou_gt70']:.1%}"
             f"  >0.8:{best_metrics['iou_gt80']:.1%}"
             f"  @ epoch {best_metrics['epoch']}")
    log.info(f"  TEST       mean_iou={test_m['mean_iou']:.4f}"
             f"  median={test_m['median_iou']:.4f}"
             f"  >0.7:{test_m['iou_gt70']:.1%}"
             f"  >0.8:{test_m['iou_gt80']:.1%}")
    delta = test_m["mean_iou"] - BASELINE_TEST_IOU
    sign  = "+" if delta >= 0 else ""
    log.info(f"  vs baseline (efficientnet_b3 25ep IoU=0.819): {sign}{delta:+.4f}")
    log.info(f"  Checkpoint: {_CKPT_PATH}")
    log.info(f"  Log:        {_LOG_FILE}")
    log.info("=" * 70)


if __name__ == "__main__":
    train()
