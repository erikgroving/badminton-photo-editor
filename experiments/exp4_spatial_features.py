"""
Experiment 4 — Spatial Feature Retention via Dual-Branch ROI-Aware Pooling
===========================================================================

Hypothesis: Global average pooling discards spatial layout, which matters for
crop prediction (WHERE the player is determines WHERE to place the crop window).

Approach (a): Dual-branch pooling on the 2D feature map
  Branch 1 — Global avg pool of the full feature map  →  C-dim vector  (global context)
  Branch 2 — ROI crop of the player-bbox region       →  C-dim vector  (player context)
  Concatenated: 2C-dim → head

ROI crop implementation: bilinear interpolation grid-sample on the feature map,
cropped to the player bbox (normalized [x1,y1,x2,y2]), then avg-pooled to 1×1.
Falls back to the global branch when no player is detected (bbox = [0,0,0,0]).

Backbone: efficientnet_b3 with global_pool='' to expose the 2D feature map.
Epochs: 10  |  batch_size: 16  |  lr: 1e-4
Log:        logs/exp4_spatial.log
Checkpoint: checkpoints/cropping_angle_efficientnet_b3_exp4.pt

Baseline comparison: efficientnet_b3 + standard global avg pool → val IoU 0.819 at 25 ep
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
BACKBONE     = "efficientnet_b3"
EPOCHS       = 10
BATCH_SIZE   = 16
LR           = 1e-4
LOG_PATH     = ROOT / "logs" / "exp4_spatial.log"
CKPT_PATH    = CHECKPOINTS_DIR / "cropping_angle_efficientnet_b3_exp4.pt"
EXTRACT_SIZE = 512
ROI_POOL_SIZE = 7      # pool the player-region feature map to 7×7 then avg → 1×1

_PLAYER_BBOX_CACHE         = ROOT / "data" / "player_bboxes.json"
_PRIMARY_PLAYER_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── logging ───────────────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_file_handler   = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
_stream_handler = logging.StreamHandler(sys.stdout)
# Force utf-8 on Windows consoles that default to cp1252
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


# ── ROI-aware feature extractor ───────────────────────────────────────────────

def _roi_pool_from_bbox(feat_map: torch.Tensor,
                        player_bbox: torch.Tensor,
                        pool_size: int = 7) -> torch.Tensor:
    """
    Bilinear-sample the feature map within each player bbox, then avg-pool to 1×1.

    feat_map:    [B, C, H, W]
    player_bbox: [B, 4]  normalized [x1, y1, x2, y2] in [0, 1]
    Returns:     [B, C]  — same as global avg pool output
    """
    B, C, H, W = feat_map.shape
    # Build sampling grids per sample
    grids = []
    for b in range(B):
        x1, y1, x2, y2 = player_bbox[b].unbind(0)
        # Clamp to valid range
        x1 = x1.clamp(0.0, 1.0)
        y1 = y1.clamp(0.0, 1.0)
        x2 = x2.clamp(x1 + 1e-4, 1.0)
        y2 = y2.clamp(y1 + 1e-4, 1.0)

        # grid_sample uses coords in [-1, 1]; map bbox into that space
        # xs ∈ [x1, x2] normalized → [-1,1] as: 2*t - 1  where t ∈ [0,1]
        xs = torch.linspace(0.0, 1.0, pool_size, device=feat_map.device)
        ys = torch.linspace(0.0, 1.0, pool_size, device=feat_map.device)
        xs_mapped = (x1 + xs * (x2 - x1)) * 2.0 - 1.0   # [pool_size]
        ys_mapped = (y1 + ys * (y2 - y1)) * 2.0 - 1.0   # [pool_size]
        # Build [pool_size, pool_size, 2] grid  (x, y)
        grid_x = xs_mapped.unsqueeze(0).expand(pool_size, -1)   # [P, P]
        grid_y = ys_mapped.unsqueeze(1).expand(-1, pool_size)   # [P, P]
        grid = torch.stack([grid_x, grid_y], dim=-1)            # [P, P, 2]
        grids.append(grid)

    grid_batch = torch.stack(grids, dim=0)   # [B, P, P, 2]
    # grid_sample: input [B,C,H,W], grid [B,P,P,2] → [B,C,P,P]
    sampled = F.grid_sample(feat_map, grid_batch,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)
    # avg pool [B, C, P, P] → [B, C]
    return sampled.mean(dim=[2, 3])


class SpatialCropRegressor(nn.Module):
    """
    Dual-branch pooling model:
      - Branch 1: global avg pool of full feature map (global context)
      - Branch 2: ROI-crop + pool of player bbox region (player context)
      - head input: branch1 ∥ branch2   (2C features)

    When no player is detected (bbox = [0,0,0,0]), branch2 == branch1 (fallback to global).
    """
    def __init__(self, backbone_name: str = "efficientnet_b3",
                 pretrained: bool = True,
                 roi_pool_size: int = 7):
        super().__init__()
        # global_pool='' → returns [B, C, H, W] feature map
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                              num_classes=0, global_pool="")
        except RuntimeError:
            self.backbone = timm.create_model(backbone_name, pretrained=False,
                                              num_classes=0, global_pool="")

        self.roi_pool_size = roi_pool_size
        C = self.backbone.num_features        # 1536 for efficientnet_b3

        # Player-bbox encoder (same as baseline: 4 → 32)
        _player_emb_dim = 32
        self.player_encoder = nn.Sequential(
            nn.Linear(4, _player_emb_dim),
            nn.ReLU(),
        )

        # Head takes 2×C (global + roi) + player_emb
        head_in = 2 * C + _player_emb_dim
        self.box_head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                union_bbox: torch.Tensor | None = None,
                player_bbox: torch.Tensor | None = None):
        """
        x:           [B, 3, H, W]
        union_bbox:  [B, 4] — used for model conditioning (encoder input)
        player_bbox: [B, 4] — used for ROI branch (primary player for spatial sampling)
        """
        feat_map = self.backbone(x)   # [B, C, fH, fW]

        # Branch 1: global avg pool
        global_feat = feat_map.mean(dim=[2, 3])    # [B, C]

        # Branch 2: ROI pool on player bbox
        # Use union_bbox for the ROI (same conditioning bbox as the baseline)
        roi_src = union_bbox if union_bbox is not None else player_bbox
        if roi_src is not None:
            has_player = (roi_src.sum(dim=1) > 0.0)  # [B] bool
            roi_feat = _roi_pool_from_bbox(feat_map, roi_src, self.roi_pool_size)
            # For samples with no player detection, use global_feat as fallback
            # so we don't introduce garbage signal from an all-zero bbox
            roi_feat = torch.where(
                has_player.unsqueeze(1).expand_as(roi_feat),
                roi_feat,
                global_feat
            )
        else:
            roi_feat = global_feat

        feats = torch.cat([global_feat, roi_feat], dim=1)   # [B, 2C]

        # Concat player embedding
        if union_bbox is not None:
            player_emb = self.player_encoder(union_bbox)     # [B, 32]
            feats = torch.cat([feats, player_emb], dim=1)   # [B, 2C+32]

        box   = self.box_head(feats)
        angle = self.angle_head(feats).squeeze(-1)
        return box, angle

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


# ── Dataset (same as train.py, no modifications to existing files) ─────────────

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
    all_pred, all_gt, all_angle_pred, all_angle_gt = [], [], [], []
    with torch.no_grad():
        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes in loader:
            box_pred, angle_pred = model(imgs.to(device), union_bboxes.to(device))
            all_pred.append(box_pred.cpu().numpy())
            all_gt.append(boxes.numpy())
            all_angle_pred.append(angle_pred.cpu().numpy())
            all_angle_gt.append(angle_norms.numpy())

    pred_arr = np.concatenate(all_pred)
    gt_arr   = np.concatenate(all_gt)
    ious     = box_iou_numpy(pred_arr, gt_arr)
    pred_deg = np.concatenate(all_angle_pred) * ANGLE_SCALE
    true_deg = np.concatenate(all_angle_gt)   * ANGLE_SCALE

    return {
        "mean_iou":      float(ious.mean()),
        "median_iou":    float(np.median(ious)),
        "iou_gt70":      float((ious >= 0.70).mean()),
        "iou_gt80":      float((ious >= 0.80).mean()),
        "angle_mae_deg": float(np.mean(np.abs(pred_deg - true_deg))),
        "n":             len(ious),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Exp4: Spatial features — dual-branch ROI-aware pooling")
    log.info(f"  Backbone:  {BACKBONE}")
    log.info(f"  Pooling:   global_avg_pool || roi_pool(player_bbox, {ROI_POOL_SIZE}x{ROI_POOL_SIZE})")
    log.info(f"  Head in:   2x{BACKBONE} features + 32 player_emb")
    log.info(f"  Epochs:    {EPOCHS}  |  batch: {BATCH_SIZE}  |  lr: {LR}")
    log.info(f"  Checkpoint: {CKPT_PATH}")
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

    # Load player bbox caches
    player_bbox_cache = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        n_cov = sum(1 for r in all_records if player_bbox_cache.get(r["raw"]) is not None)
        log.info(f"  Union bbox cache: {len(player_bbox_cache):,} entries ({n_cov}/{len(all_records)} covered)")
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

    model = SpatialCropRegressor(
        backbone_name=BACKBONE,
        pretrained=True,
        roi_pool_size=ROI_POOL_SIZE,
    ).to(device)

    C = model.backbone.num_features
    log.info(f"  Backbone features: {C}  |  head_in: {2*C + 32}")

    # Data config from timm (for input_size / normalization)
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

    criterion = CropLoss(alpha=0.5, angle_weight=0.25,
                         player_weight=0.5, player_margin=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_iou     = -1.0
    best_metrics = {}
    start_epoch  = 1

    # Resume from checkpoint if it exists and resume flag is set
    import sys as _sys
    _resume = "--resume" in _sys.argv
    if _resume and CKPT_PATH.exists():
        ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        model.to(device)
        start_epoch  = ck["epoch"] + 1
        best_iou     = ck["metrics"]["mean_iou"]
        best_metrics = dict(ck["metrics"], backbone=BACKBONE, epoch=ck["epoch"])
        remaining    = EPOCHS - (start_epoch - 1)
        log.info(f"  Resumed from ep{ck['epoch']} (val IoU={best_iou:.4f}); "
                 f"continuing for {remaining} more epochs")
        if remaining > 0:
            sched = CosineAnnealingLR(optimizer, T_max=max(remaining, 1))

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes in tqdm(
                train_loader, desc=f"ep{epoch}/{EPOCHS}", leave=False):
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            angle_norms    = angle_norms.to(device)
            union_bboxes   = union_bboxes.to(device)
            primary_bboxes = primary_bboxes.to(device)

            optimizer.zero_grad()
            box_pred, angle_pred = model(imgs, union_bboxes)
            loss = criterion(box_pred, boxes,
                             angle_pred=angle_pred,
                             angle_target=angle_norms,
                             player_bbox=primary_bboxes)
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
            f"  angle_mae={val_m['angle_mae_deg']:.2f}deg"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m, backbone=BACKBONE, epoch=epoch)
            torch.save({
                "epoch":          epoch,
                "backbone":       BACKBONE,
                "model_state":    model.state_dict(),
                "metrics":        val_m,
                "input_size":     input_size,
                "norm_mean":      norm_mean,
                "norm_std":       norm_std,
                "angle_scale":    ANGLE_SCALE,
                "exp":            "exp4_spatial_roi_pooling",
                "roi_pool_size":  ROI_POOL_SIZE,
            }, CKPT_PATH)
            log.info(f"    [BEST] Saved (mean_iou={best_iou:.4f})")

    # Final test evaluation using best checkpoint
    ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = _evaluate(model, test_loader, device)
    best_metrics["test_metrics"] = test_m

    log.info("")
    log.info("=" * 70)
    log.info(f"[FINAL] Best val  mean_iou={best_metrics['mean_iou']:.4f}  "
             f"at epoch {best_metrics['epoch']}")
    log.info(f"[FINAL] Test      mean_iou={test_m['mean_iou']:.4f}"
             f"  median={test_m['median_iou']:.4f}"
             f"  >0.7:{test_m['iou_gt70']:.1%}"
             f"  >0.8:{test_m['iou_gt80']:.1%}"
             f"  angle_mae={test_m['angle_mae_deg']:.2f}deg")
    log.info(f"Baseline (efficientnet_b3, 25ep): val IoU = 0.819")
    diff = best_metrics["mean_iou"] - 0.819
    log.info(f"Delta vs baseline: {diff:+.4f}"
             f"  ({'BEAT' if diff > 0 else 'BELOW'} baseline)")
    log.info("=" * 70)

    return best_metrics


if __name__ == "__main__":
    main()
