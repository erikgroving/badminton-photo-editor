"""
Experiment 4 — Spatial Backbone Features with Dual-Pool Conditioning
=====================================================================

Hypothesis: Global average pooling discards spatial information. By combining
BOTH global avg-pool AND global max-pool on the spatial feature map, the model
retains richer per-region information: avg captures distributed context while
max captures the strongest activations anywhere in the map. Together they give
the box head more spatial signal than avg-pool alone.

Architecture:
  backbone: efficientnet_b3, global_pool='' → [B, C, H, W] feature map
  Branch 1: global avg pool → [B, C]
  Branch 2: global max pool → [B, C]
  concat → [B, 2C]
  Linear(2C, 512) bottleneck → [B, 512]         (to keep head manageable)
  player_encoder(bbox[4]) → [B, 32]
  box_head: Linear(512+32, 256) → ReLU → Dropout(0.3) → Linear(256, 4) → Sigmoid

Checkpoint tag: _exp4_spatial   (unique, never used by other experiments)
Checkpoint:     checkpoints/cropping_efficientnet_b3_exp4_spatial.pt

Baseline: efficientnet_b3 + global avg pool only → test IoU = 0.819 @ 25 ep

Training: 10 epochs, batch_size=16, lr=1e-4, cosine LR schedule
Log:      logs/exp4_spatial.log
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

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import ANGLE_SCALE, CropLoss, box_iou_numpy

# ── config ─────────────────────────────────────────────────────────────────────
BACKBONE        = "efficientnet_b3"
EPOCHS          = 10
BATCH_SIZE      = 16
LR              = 1e-4
CKPT_TAG        = "_exp4_spatial"
LOG_PATH        = ROOT / "logs" / "exp4_spatial.log"
CKPT_PATH       = CHECKPOINTS_DIR / f"cropping_{BACKBONE}{CKPT_TAG}.pt"
EXTRACT_SIZE    = 512
BOTTLENECK_DIM  = 512    # Linear(2C → 512) before head to keep param count reasonable
PLAYER_EMB_DIM  = 32

_PLAYER_BBOX_CACHE         = ROOT / "data" / "player_bboxes.json"
_PRIMARY_PLAYER_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── logging ───────────────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_file_handler   = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
_stream_handler = logging.StreamHandler(sys.stdout)
if hasattr(_stream_handler.stream, "reconfigure"):
    try:
        _stream_handler.stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_file_handler.setFormatter(_fmt)
_stream_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
log = logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────

class DualPoolCropRegressor(nn.Module):
    """
    Spatial feature model using dual-pool (avg + max) on the 2D feature map.

    Unlike the baseline's single global_avg_pool, this model:
      1. Retains the full [B, C, H, W] spatial map (no pooling in backbone).
      2. Applies BOTH avg-pool and max-pool to capture different statistics.
      3. Passes through a bottleneck Linear before the head.
      4. Conditions on the player bbox embedding (same as baseline).

    No angle head — box prediction only (simpler, avoids confounding factors).
    """
    def __init__(self, backbone_name: str = "efficientnet_b3",
                 pretrained: bool = True,
                 bottleneck_dim: int = BOTTLENECK_DIM,
                 player_emb_dim: int = PLAYER_EMB_DIM):
        super().__init__()
        # No global pooling → returns [B, C, H, W]
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                              num_classes=0, global_pool="")
        except RuntimeError:
            self.backbone = timm.create_model(backbone_name, pretrained=False,
                                              num_classes=0, global_pool="")

        C = self.backbone.num_features   # 1536 for efficientnet_b3

        # Bottleneck: 2C → bottleneck_dim  (reduces head param count vs 2C→256 directly)
        self.bottleneck = nn.Sequential(
            nn.Linear(2 * C, bottleneck_dim),
            nn.ReLU(),
        )

        # Player bbox encoder (same architecture as baseline)
        self.player_encoder = nn.Sequential(
            nn.Linear(4, player_emb_dim),
            nn.ReLU(),
        )

        # Box head: (bottleneck + player_emb) → 4 coords
        head_in = bottleneck_dim + player_emb_dim
        self.box_head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor,
                player_bbox: torch.Tensor | None = None):
        """
        x:           [B, 3, H, W]
        player_bbox: [B, 4] normalized bbox for conditioning

        Returns: box [B, 4]
        """
        feat_map = self.backbone(x)           # [B, C, fH, fW]

        avg_feat = feat_map.mean(dim=[2, 3])  # [B, C]
        max_feat = feat_map.amax(dim=[2, 3])  # [B, C]

        dual = torch.cat([avg_feat, max_feat], dim=1)   # [B, 2C]
        spatial_ctx = self.bottleneck(dual)              # [B, bottleneck_dim]

        if player_bbox is not None:
            player_emb = self.player_encoder(player_bbox)       # [B, player_emb_dim]
            feats = torch.cat([spatial_ctx, player_emb], dim=1) # [B, bottleneck + emb]
        else:
            feats = spatial_ctx

        return self.box_head(feats)   # [B, 4]

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Dataset ────────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    def __init__(self, records, transform, hflip=False,
                 player_bbox_cache=None, primary_bbox_cache=None):
        self.records            = records
        self.transform          = transform
        self.hflip              = hflip
        self.player_bbox_cache  = player_bbox_cache  or {}
        self.primary_bbox_cache = primary_bbox_cache or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)
        box = list(r["box"])
        angle_norm = r.get("angle_deg", 0.0) / ANGLE_SCALE

        ub = self.player_bbox_cache.get(r["raw"])
        union_bbox = list(ub) if ub is not None else [0.0, 0.0, 0.0, 0.0]

        pb = self.primary_bbox_cache.get(r["raw"])
        primary_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]

        if self.hflip and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x1, y1, x2, y2 = box
            box = [1.0 - x2, y1, 1.0 - x1, y2]
            if ub is not None:
                ux1, uy1, ux2, uy2 = union_bbox
                union_bbox = [1.0 - ux2, uy1, 1.0 - ux1, uy2]
            if pb is not None:
                px1, py1, px2, py2 = primary_bbox
                primary_bbox = [1.0 - px2, py1, 1.0 - px1, py2]

        img_t          = self.transform(img)
        box_t          = torch.tensor(box,          dtype=torch.float32)
        angle_t        = torch.tensor(angle_norm,   dtype=torch.float32)
        union_bbox_t   = torch.tensor(union_bbox,   dtype=torch.float32)
        primary_bbox_t = torch.tensor(primary_bbox, dtype=torch.float32)
        return img_t, box_t, angle_t, union_bbox_t, primary_bbox_t


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes in loader:
            box_pred = model(imgs.to(device), union_bboxes.to(device))
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Exp4 Spatial: Dual-Pool (avg+max) Spatial Feature Conditioning")
    log.info(f"  Backbone:      {BACKBONE} (global_pool='')")
    log.info(f"  Dual pool:     avg_pool([B,C,H,W]) || max_pool([B,C,H,W]) → [B,2C]")
    log.info(f"  Bottleneck:    Linear(2C, {BOTTLENECK_DIM})")
    log.info(f"  Player emb:    Linear(4, {PLAYER_EMB_DIM})")
    log.info(f"  Epochs:        {EPOCHS}  |  batch: {BATCH_SIZE}  |  lr: {LR}")
    log.info(f"  Ckpt tag:      {CKPT_TAG}")
    log.info(f"  Checkpoint:    {CKPT_PATH}")
    log.info(f"  Log:           {LOG_PATH}")
    log.info("=" * 70)

    if not CROP_GT_FILE.exists():
        log.error(f"GT file not found: {CROP_GT_FILE}")
        sys.exit(1)

    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    log.info(f"  Data: train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    # Load bbox caches
    player_bbox_cache = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        n_cov = sum(1 for r in all_records if player_bbox_cache.get(r["raw"]) is not None)
        log.info(f"  Union bbox cache:   {len(player_bbox_cache):,} entries ({n_cov}/{len(all_records)} covered)")
    else:
        log.warning(f"  No player bbox cache at {_PLAYER_BBOX_CACHE}")

    primary_bbox_cache = {}
    if _PRIMARY_PLAYER_BBOX_CACHE.exists():
        with open(_PRIMARY_PLAYER_BBOX_CACHE) as fh:
            primary_bbox_cache = json.load(fh)
        n_prim = sum(1 for r in all_records if primary_bbox_cache.get(r["raw"]) is not None)
        log.info(f"  Primary bbox cache: {len(primary_bbox_cache):,} entries ({n_prim}/{len(all_records)} covered)")
    else:
        primary_bbox_cache = player_bbox_cache
        log.info("  Primary bbox not found — using union bbox for clipping penalty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")

    model = DualPoolCropRegressor(
        backbone_name=BACKBONE,
        pretrained=True,
        bottleneck_dim=BOTTLENECK_DIM,
        player_emb_dim=PLAYER_EMB_DIM,
    ).to(device)

    n_params = count_parameters(model)
    C = model.backbone.num_features
    log.info(f"  Backbone features: C={C}")
    log.info(f"  Head input:        2×{C} → {BOTTLENECK_DIM} (bottleneck) + {PLAYER_EMB_DIM} (player) → 256 → 4")
    log.info(f"  Trainable params:  {n_params:,}")
    # Baseline CropRegressor: backbone + Linear(C,256) + Linear(256,4) + player_encoder + angle head
    baseline_head_params = (C * 256 + 256) + (256 * 4 + 4)  # rough estimate of head only
    log.info(f"  (Baseline head adds ~{baseline_head_params:,} params on top of same backbone)")

    # Data config from timm
    data_cfg   = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_cfg.get("input_size", (3, 300, 300))[1]
    norm_mean  = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    norm_std   = tuple(data_cfg.get("std",  _IMAGENET_STD))
    log.info(f"  Input size: {input_size}x{input_size}")

    tf_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(list(norm_mean), list(norm_std)),
    ])
    tf_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(list(norm_mean), list(norm_std)),
    ])

    nw = min(4, BATCH_SIZE)
    train_loader = DataLoader(
        CropDataset(train_recs, tf_train, hflip=True,
                    player_bbox_cache=player_bbox_cache,
                    primary_bbox_cache=primary_bbox_cache),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        CropDataset(val_recs, tf_val,
                    player_bbox_cache=player_bbox_cache,
                    primary_bbox_cache=primary_bbox_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        CropDataset(test_recs, tf_val,
                    player_bbox_cache=player_bbox_cache,
                    primary_bbox_cache=primary_bbox_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )

    # Box-only loss (no angle head)
    criterion = CropLoss(alpha=0.5, angle_weight=0.0,
                         player_weight=0.5, player_margin=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_iou     = -1.0
    best_metrics = {}
    best_epoch   = -1

    log.info("")
    log.info("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes in tqdm(
                train_loader, desc=f"ep{epoch}/{EPOCHS}", leave=False):
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            union_bboxes   = union_bboxes.to(device)
            primary_bboxes = primary_bboxes.to(device)

            optimizer.zero_grad()
            box_pred = model(imgs, union_bboxes)
            loss = criterion(box_pred, boxes, player_bbox=primary_bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        sched.step()

        val_m = _evaluate(model, val_loader, device)
        log.info(
            f"  ep{epoch:02d}"
            f"  loss={total_loss/len(train_loader):.4f}"
            f"  [val] mean_iou={val_m['mean_iou']:.4f}"
            f"  med={val_m['median_iou']:.4f}"
            f"  >0.7:{val_m['iou_gt70']:.1%}"
            f"  >0.8:{val_m['iou_gt80']:.1%}"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m)
            best_epoch   = epoch
            torch.save({
                "epoch":         epoch,
                "backbone":      BACKBONE,
                "ckpt_tag":      CKPT_TAG,
                "model_state":   model.state_dict(),
                "metrics":       val_m,
                "input_size":    input_size,
                "norm_mean":     norm_mean,
                "norm_std":      norm_std,
                "bottleneck_dim": BOTTLENECK_DIM,
                "player_emb_dim": PLAYER_EMB_DIM,
                "exp":           "exp4_spatial_dual_pool",
            }, CKPT_PATH)
            log.info(f"    [BEST] Saved checkpoint (mean_iou={best_iou:.4f})")

    # Final test evaluation using best val checkpoint
    log.info("")
    log.info("Loading best checkpoint for test evaluation...")
    ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = _evaluate(model, test_loader, device)

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    log.info(f"  Backbone:       {BACKBONE} + dual-pool (avg+max) + bottleneck({BOTTLENECK_DIM})")
    log.info(f"  Trainable params: {n_params:,}")
    log.info(f"  Best val epoch:   {best_epoch}/{EPOCHS}")
    log.info(f"  Best val IoU:     {best_iou:.4f}")
    log.info("")
    log.info(f"  [TEST] mean_iou   = {test_m['mean_iou']:.4f}")
    log.info(f"  [TEST] median_iou = {test_m['median_iou']:.4f}")
    log.info(f"  [TEST] >0.7 iou   = {test_m['iou_gt70']:.1%}")
    log.info(f"  [TEST] >0.8 iou   = {test_m['iou_gt80']:.1%}")
    log.info("")
    log.info(f"  Baseline (efficientnet_b3, avg-pool only, 25ep): test IoU = 0.819")
    delta = test_m["mean_iou"] - 0.819
    log.info(f"  Delta vs baseline (10ep vs 25ep):  {delta:+.4f}"
             f"  ({'BEAT' if delta > 0 else 'below'} baseline)")
    log.info("=" * 70)

    return {
        "backbone":      BACKBONE,
        "ckpt_tag":      CKPT_TAG,
        "n_params":      n_params,
        "best_val_iou":  best_iou,
        "best_epoch":    best_epoch,
        "test_metrics":  test_m,
    }


if __name__ == "__main__":
    main()
