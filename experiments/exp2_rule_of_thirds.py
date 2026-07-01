"""
Experiment 2: Rule-of-Thirds (RoT) conditioning for the crop regressor.

Hypothesis: Jay Ma places the primary player near a rule-of-thirds intersection
inside the final cropped frame.  If true, we can condition the model on where
the player sits relative to RoT guidelines in the RAW frame and optionally add
an auxiliary loss that rewards placing the player at a RoT intersection in the
predicted crop.

Changes versus the baseline (model.py / train.py):
  - New RoT feature vector (7-dim) computed from the primary player bbox:
        [dist_cx_to_nearest_rot_vertical,
         dist_cy_to_nearest_rot_horizontal,
         signed distance to left RoT vertical,
         signed distance to right RoT vertical,
         signed distance to top RoT horizontal,
         signed distance to bottom RoT horizontal,
         sqrt(player_area)]               (proxy for shot distance)
  - player_encoder now takes 4 + 7 = 11 inputs instead of 4
  - Optional RoT auxiliary loss: rewards placing the player center at a RoT
    intersection inside the *predicted* crop (weight=0.1)

Checkpoint : checkpoints/cropping_angle_efficientnet_b3_exp2.pt
Log        : logs/exp2_rot.log

Usage:
    python experiments/exp2_rule_of_thirds.py
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

# -- project root on sys.path -------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import ANGLE_SCALE, box_iou_numpy, _box_iou_loss

# -- constants ----------------------------------------------------------------
BACKBONE       = "efficientnet_b3"
EPOCHS         = 10
BATCH_SIZE     = 16
LR             = 1e-4
ROT_LOSS_W     = 0.1   # weight for the RoT auxiliary loss term
PLAYER_LOSS_W  = 0.5   # keep the existing player-coverage penalty
SMOOTH_L1_W    = 0.5   # weight of SmoothL1 vs (1-IoU) in box loss

LOG_PATH  = ROOT / "logs"   / "exp2_rot.log"
CKPT_PATH = CHECKPOINTS_DIR / "cropping_angle_efficientnet_b3_exp2.pt"

_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_UNION_BBOX_CACHE   = ROOT / "data" / "player_bboxes.json"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_EXTRACT_SIZE  = 512

# -- RoT intersections --------------------------------------------------------
ROT_LINES = (1 / 3, 2 / 3)          # verticals AND horizontals


# -----------------------------------------------------------------------------
#  Analysis helpers
# -----------------------------------------------------------------------------

def _player_in_crop_coords(player_bbox, crop_box):
    """
    Given player bbox [px1,py1,px2,py2] and crop box [cx1,cy1,cx2,cy2], both in
    raw-frame coordinates [0,1], return the player center expressed in crop-local
    coordinates [0,1] (0 = left/top edge of crop, 1 = right/bottom edge).
    Returns None if player bbox is all zeros or crop has zero area.
    """
    px1, py1, px2, py2 = player_bbox
    cx1, cy1, cx2, cy2 = crop_box
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None
    if px1 == py1 == px2 == py2 == 0.0:
        return None
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    local_x = (pcx - cx1) / cw
    local_y = (pcy - cy1) / ch
    return local_x, local_y


def _dist_to_nearest_rot(v: float) -> float:
    """Distance from v to the nearest RoT line (0.333 or 0.667)."""
    return min(abs(v - t) for t in ROT_LINES)


def _rot_quadrant(cx: float, cy: float) -> int:
    """
    Which of 4 RoT regions is the center in?
    (left/right) x (top/bottom), treating the midpoint 0.5 as the split.
    Returns 0-3.
    """
    col = 0 if cx < 0.5 else 1
    row = 0 if cy < 0.5 else 1
    return row * 2 + col


def analyze_rot(gt_records: list, primary_cache: dict) -> dict:
    """
    For every GT record that has a primary player bbox, compute where the player
    center sits inside the GT crop in crop-local coords.  Report clustering around
    RoT intersections.
    """
    local_xs, local_ys = [], []
    skipped = 0

    for r in gt_records:
        pb = primary_cache.get(r["raw"])
        if pb is None:
            skipped += 1
            continue
        result = _player_in_crop_coords(pb, r["box"])
        if result is None:
            skipped += 1
            continue
        local_xs.append(result[0])
        local_ys.append(result[1])

    n = len(local_xs)
    if n == 0:
        return {"error": "no samples with both GT crop and primary bbox", "skipped": skipped}

    lx = np.array(local_xs)
    ly = np.array(local_ys)

    # distance to nearest RoT line
    dx = np.array([_dist_to_nearest_rot(v) for v in lx])
    dy = np.array([_dist_to_nearest_rot(v) for v in ly])

    # fraction within 0.10 of a RoT line
    frac_x_near  = float((dx < 0.10).mean())
    frac_y_near  = float((dy < 0.10).mean())
    frac_xy_near = float(((dx < 0.10) & (dy < 0.10)).mean())   # near an intersection

    # quadrant histogram (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    quads = np.array([_rot_quadrant(x, y) for x, y in zip(lx, ly)])
    quad_counts = {
        "top-left":     int((quads == 0).sum()),
        "top-right":    int((quads == 1).sum()),
        "bottom-left":  int((quads == 2).sum()),
        "bottom-right": int((quads == 3).sum()),
    }

    return {
        "n":               n,
        "skipped":         skipped,
        "mean_local_cx":   float(lx.mean()),
        "mean_local_cy":   float(ly.mean()),
        "std_local_cx":    float(lx.std()),
        "std_local_cy":    float(ly.std()),
        "mean_dx_to_rot":  float(dx.mean()),
        "mean_dy_to_rot":  float(dy.mean()),
        "frac_x_near_rot": frac_x_near,
        "frac_y_near_rot": frac_y_near,
        "frac_near_intersection": frac_xy_near,
        "quadrant_counts": quad_counts,
    }


# -----------------------------------------------------------------------------
#  RoT feature extractor
# -----------------------------------------------------------------------------

def compute_rot_features(player_bbox: torch.Tensor) -> torch.Tensor:
    """
    player_bbox: [B, 4] -- (x1, y1, x2, y2) in raw-frame coords [0, 1]
    Returns:     [B, 7]
      [0] dist cx -> nearest RoT vertical  (0 = on a vertical RoT line)
      [1] dist cy -> nearest RoT horizontal
      [2] signed dist: cx - 0.333  (negative = left of left RoT line)
      [3] signed dist: cx - 0.667  (positive = right of right RoT line)
      [4] signed dist: cy - 0.333
      [5] signed dist: cy - 0.667
      [6] sqrt(player_area)  -- proxy for distance to player
    """
    x1, y1, x2, y2 = player_bbox.unbind(1)   # each [B]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    r1, r2 = 1 / 3, 2 / 3
    dx = torch.minimum(torch.abs(cx - r1), torch.abs(cx - r2))
    dy = torch.minimum(torch.abs(cy - r1), torch.abs(cy - r2))

    # signed distance features (continuous positional signal)
    sig_cx_left  = cx - r1
    sig_cx_right = cx - r2
    sig_cy_top   = cy - r1
    sig_cy_bot   = cy - r2

    area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).sqrt()

    return torch.stack([dx, dy, sig_cx_left, sig_cx_right,
                        sig_cy_top, sig_cy_bot, area], dim=1)   # [B, 7]


# -----------------------------------------------------------------------------
#  Model
# -----------------------------------------------------------------------------

def build_rot_model(backbone: str = BACKBONE, pretrained: bool = True) -> nn.Module:
    backbone_model = timm.create_model(backbone, pretrained=pretrained,
                                       num_classes=0, global_pool="avg")
    in_features    = backbone_model.num_features
    player_emb_dim = 64          # wider than baseline (4->32) because input is 11-dim
    head_in        = in_features + player_emb_dim

    class RoTCropRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_model
            # Accepts union bbox (4) + RoT features (7) = 11 inputs
            self.player_encoder = nn.Sequential(
                nn.Linear(4 + 7, player_emb_dim),
                nn.ReLU(),
                nn.Linear(player_emb_dim, player_emb_dim),
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
                    union_bbox: torch.Tensor,
                    primary_bbox: torch.Tensor) -> torch.Tensor:
            feats    = self.backbone(x)
            rot_feat = compute_rot_features(primary_bbox)          # [B, 7]
            cond     = torch.cat([union_bbox, rot_feat], dim=1)    # [B, 11]
            feats    = torch.cat([feats, self.player_encoder(cond)], dim=1)
            return self.box_head(feats)

        def set_grad_checkpointing(self, enable: bool = True) -> None:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=enable)

    return RoTCropRegressor()


# -----------------------------------------------------------------------------
#  RoT auxiliary loss
# -----------------------------------------------------------------------------

def rot_alignment_loss(box_pred: torch.Tensor,
                       primary_bbox: torch.Tensor) -> torch.Tensor:
    """
    For samples where a primary player was detected, reward placing the player
    center near a RoT intersection inside the predicted crop.

    Penalises distance of the player center (expressed in crop-local coords)
    from the nearest RoT intersection (1/3, 1/3), (1/3, 2/3), (2/3, 1/3),
    (2/3, 2/3).

    Returns a scalar loss term (0.0 if no sample has a detected player).
    """
    has_player = primary_bbox.sum(dim=1) > 0.0      # [B]
    if not has_player.any():
        return box_pred.sum() * 0.0                 # zero but keeps gradient graph

    pb = primary_bbox[has_player]
    pp = box_pred[has_player]

    px1, py1, px2, py2 = pb.unbind(1)
    cx1, cy1, cx2, cy2 = pp.unbind(1)

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    cw  = (cx2 - cx1).clamp(min=1e-6)
    ch  = (cy2 - cy1).clamp(min=1e-6)

    # local coords of player center inside predicted crop [0,1]
    local_x = ((pcx - cx1) / cw).clamp(0.0, 1.0)
    local_y = ((pcy - cy1) / ch).clamp(0.0, 1.0)

    # distance to nearest RoT intersection in each axis
    r1, r2  = 1 / 3, 2 / 3
    dx      = torch.minimum(torch.abs(local_x - r1), torch.abs(local_x - r2))
    dy      = torch.minimum(torch.abs(local_y - r1), torch.abs(local_y - r2))

    return (dx + dy).mean()


# -----------------------------------------------------------------------------
#  Loss
# -----------------------------------------------------------------------------

class RoTCropLoss(nn.Module):
    def __init__(self, alpha=SMOOTH_L1_W, player_weight=PLAYER_LOSS_W,
                 rot_weight=ROT_LOSS_W):
        super().__init__()
        self.alpha         = alpha
        self.player_weight = player_weight
        self.rot_weight    = rot_weight

    def forward(self, box_pred, box_target, primary_bbox):
        sl1      = nn.functional.smooth_l1_loss(box_pred, box_target)
        iou      = _box_iou_loss(box_pred, box_target)
        box_loss = self.alpha * sl1 + (1.0 - self.alpha) * iou
        total    = box_loss

        # Player coverage penalty (hinge; keeps player fully in crop)
        if self.player_weight > 0:
            has_player = primary_bbox.sum(dim=1) > 0.0
            if has_player.any():
                # Use masking via multiplication to avoid boolean-index OOM on CUDA
                mask = has_player.float().unsqueeze(1)   # [B, 1]
                px1, py1, px2, py2 = primary_bbox.unbind(1)
                cx1, cy1, cx2, cy2 = box_pred.unbind(1)
                clip_per_sample = (
                    (cx1 - px1).clamp(min=0.0) +
                    (cy1 - py1).clamp(min=0.0) +
                    (px2 - cx2).clamp(min=0.0) +
                    (py2 - cy2).clamp(min=0.0)
                )                                       # [B]
                n_players = has_player.float().sum().clamp(min=1.0)
                clip_loss = (clip_per_sample * has_player.float()).sum() / n_players
                total = total + self.player_weight * clip_loss

        # RoT alignment auxiliary loss
        if self.rot_weight > 0:
            rot_loss = rot_alignment_loss(box_pred, primary_bbox)
            total    = total + self.rot_weight * rot_loss

        return total


# -----------------------------------------------------------------------------
#  Dataset
# -----------------------------------------------------------------------------

class RoTCropDataset(Dataset):
    def __init__(self, records, transform, hflip=False,
                 union_cache=None, primary_cache=None):
        self.records        = records
        self.transform      = transform
        self.hflip          = hflip
        self.union_cache    = union_cache   or {}
        self.primary_cache  = primary_cache or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=_EXTRACT_SIZE)
        box = list(r["box"])

        ub = self.union_cache.get(r["raw"])
        union_bbox = list(ub) if ub is not None else [0.0, 0.0, 0.0, 0.0]

        pb = self.primary_cache.get(r["raw"])
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

        return (
            self.transform(img),
            torch.tensor(box,          dtype=torch.float32),
            torch.tensor(union_bbox,   dtype=torch.float32),
            torch.tensor(primary_bbox, dtype=torch.float32),
        )


# -----------------------------------------------------------------------------
#  Evaluation
# -----------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for imgs, boxes, union_bboxes, primary_bboxes in loader:
            box_pred = model(imgs.to(device),
                             union_bboxes.to(device),
                             primary_bboxes.to(device))
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


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    # -- logging --------------------------------------------------------------
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_PATH), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Fix Windows cp1252 stdout encoding
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Experiment 2: Rule-of-Thirds conditioning")
    log.info(f"  backbone={BACKBONE}  epochs={EPOCHS}  bs={BATCH_SIZE}  lr={LR}")
    log.info(f"  rot_loss_weight={ROT_LOSS_W}  player_loss_weight={PLAYER_LOSS_W}")
    log.info("=" * 70)

    # -- load GT --------------------------------------------------------------
    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    log.info(f"GT split: train={len(train_recs):,}  val={len(val_recs):,}  "
             f"test={len(test_recs):,}")

    # -- load bbox caches -----------------------------------------------------
    union_cache, primary_cache = {}, {}
    if _UNION_BBOX_CACHE.exists():
        with open(_UNION_BBOX_CACHE) as fh:
            union_cache = json.load(fh)
        log.info(f"Union bbox cache: {len(union_cache):,} entries")
    else:
        log.warning(f"Union bbox cache not found: {_UNION_BBOX_CACHE}")

    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            primary_cache = json.load(fh)
        log.info(f"Primary bbox cache: {len(primary_cache):,} entries")
    else:
        primary_cache = union_cache
        log.warning("Primary bbox cache not found - using union bbox as fallback")

    # -- RoT analysis ---------------------------------------------------------
    log.info("")
    log.info("-" * 60)
    log.info("RoT ANALYSIS (all splits)")
    analysis = analyze_rot(all_records, primary_cache)
    log.info(f"  Samples analysed    : {analysis.get('n', 0):,}  "
             f"(skipped {analysis.get('skipped', 0):,} - no bbox)")
    if "error" not in analysis:
        log.info(f"  Mean player-in-crop cx : {analysis['mean_local_cx']:.3f}  "
                 f"(+/-{analysis['std_local_cx']:.3f})")
        log.info(f"  Mean player-in-crop cy : {analysis['mean_local_cy']:.3f}  "
                 f"(+/-{analysis['std_local_cy']:.3f})")
        log.info(f"  Mean dx to RoT vertical   : {analysis['mean_dx_to_rot']:.3f}")
        log.info(f"  Mean dy to RoT horizontal : {analysis['mean_dy_to_rot']:.3f}")
        log.info(f"  Frac cx within 0.10 of RoT vertical  : "
                 f"{analysis['frac_x_near_rot']:.1%}")
        log.info(f"  Frac cy within 0.10 of RoT horizontal: "
                 f"{analysis['frac_y_near_rot']:.1%}")
        log.info(f"  Frac near an RoT intersection (both)  : "
                 f"{analysis['frac_near_intersection']:.1%}")
        log.info(f"  Quadrant distribution:")
        for name, cnt in analysis["quadrant_counts"].items():
            pct = cnt / analysis["n"] * 100
            log.info(f"    {name:15s}: {cnt:4d}  ({pct:.1f}%)")
        rot_support = analysis["frac_near_intersection"] > 0.25
        log.info("")
        if rot_support:
            log.info("  VERDICT: RoT hypothesis SUPPORTED - "
                     f"{analysis['frac_near_intersection']:.1%} of player centers "
                     "fall near an RoT intersection (>25% threshold)")
        else:
            log.info("  VERDICT: RoT hypothesis WEAK - "
                     f"only {analysis['frac_near_intersection']:.1%} near intersections "
                     "(<=25% threshold)")
    log.info("-" * 60)
    log.info("")

    # -- model & transforms ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = build_rot_model(backbone=BACKBONE, pretrained=True).to(device)

    data_cfg   = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_cfg.get("input_size", (3, 224, 224))[1]
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
        RoTCropDataset(train_recs, tf_train, hflip=True,
                       union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        RoTCropDataset(val_recs, tf_val,
                       union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        RoTCropDataset(test_recs, tf_val,
                       union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )

    criterion = RoTCropLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_iou     = -1.0
    best_metrics: dict = {}
    val_iou_per_epoch = []

    log.info(f"Training {BACKBONE} for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, boxes, union_bboxes, primary_bboxes in tqdm(
                train_loader, desc=f"ep{epoch}/{EPOCHS}", leave=False):
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            union_bboxes   = union_bboxes.to(device)
            primary_bboxes = primary_bboxes.to(device)

            optimizer.zero_grad()
            box_pred = model(imgs, union_bboxes, primary_bboxes)
            loss     = criterion(box_pred, boxes, primary_bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        sched.step()

        val_m = evaluate(model, val_loader, device)
        val_iou_per_epoch.append(val_m["mean_iou"])
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
            best_metrics = dict(val_m, backbone=BACKBONE, epoch=epoch)
            torch.save({
                "epoch":        epoch,
                "backbone":     BACKBONE,
                "model_state":  model.state_dict(),
                "metrics":      val_m,
                "input_size":   input_size,
                "norm_mean":    norm_mean,
                "norm_std":     norm_std,
                "rot_loss_w":   ROT_LOSS_W,
            }, str(CKPT_PATH))
            log.info(f"    [SAVED] best val IoU={best_iou:.4f}")

    # -- test evaluation -------------------------------------------------------
    ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = evaluate(model, test_loader, device)
    log.info("")
    log.info(f"TEST  mean_iou={test_m['mean_iou']:.4f}"
             f"  median={test_m['median_iou']:.4f}"
             f"  >0.7:{test_m['iou_gt70']:.1%}"
             f"  >0.8:{test_m['iou_gt80']:.1%}")

    # -- summary ---------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info(f"  Best val IoU : {best_iou:.4f}  (epoch {best_metrics.get('epoch')})")
    log.info(f"  Test  IoU    : {test_m['mean_iou']:.4f}")
    log.info(f"  Val IoU per epoch:")
    for ep, iou in enumerate(val_iou_per_epoch, 1):
        marker = " <--best" if iou == best_iou else ""
        log.info(f"    ep{ep:02d}: {iou:.4f}{marker}")
    log.info(f"  Checkpoint   : {CKPT_PATH}")
    log.info(f"  Log          : {LOG_PATH}")
    log.info("=" * 70)

    if "error" not in analysis:
        log.info("")
        log.info("RoT hypothesis verdict (recap):")
        log.info(f"  {analysis['frac_near_intersection']:.1%} of player centers "
                 f"are within 0.10 of an RoT intersection in crop-local coords.")
        dom_quad = max(analysis["quadrant_counts"],
                       key=lambda k: analysis["quadrant_counts"][k])
        log.info(f"  Dominant quadrant: {dom_quad} "
                 f"({analysis['quadrant_counts'][dom_quad]} / {analysis['n']})")
        log.info(f"  Mean player cx in crop: {analysis['mean_local_cx']:.3f}  "
                 f"cy: {analysis['mean_local_cy']:.3f}")

    return {
        "analysis":          analysis,
        "val_iou_per_epoch": val_iou_per_epoch,
        "best_val_iou":      best_iou,
        "test_metrics":      test_m,
    }


if __name__ == "__main__":
    main()
