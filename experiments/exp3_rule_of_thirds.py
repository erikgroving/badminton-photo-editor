"""
Experiment 3 (Rule-of-Thirds variant): 12-dim RoT conditioning + auxiliary loss.

Hypothesis: Jay Ma unconsciously places the primary player at one of the 4
rule-of-thirds intersections inside the final crop.  We condition the model on
a richer 12-dimensional RoT feature vector computed from the primary player
bbox and add a lightweight auxiliary loss that rewards placing the player at an
RoT intersection inside the predicted crop.

Feature vector (12 dims, computed from primary_bbox analytically):
  [0-3]  Distances from player center to each of 4 RoT intersections:
             (1/3,1/3), (2/3,1/3), (1/3,2/3), (2/3,2/3)          -> [4]
  [4-6]  One-hot: which RoT vertical zone is player center in?
             left (<1/3), center (1/3..2/3), right (>2/3)          -> [3]
  [7-9]  One-hot: which RoT horizontal zone?
             top (<1/3), middle (1/3..2/3), bottom (>2/3)          -> [3]
  [10]   Player width  (x2-x1)                                     -> [1]
  [11]   Player height (y2-y1)                                     -> [1]
         Total: 12

Model changes vs baseline:
  - player_encoder: Linear(12 -> 48) + ReLU  (input is primary_bbox-derived)
  - Backbone features concat with player embedding -> box head (unchanged)
  - RoT auxiliary loss weight = 0.1 (on top of the standard combined loss)

Settings:
  - backbone: efficientnet_b3
  - 10 epochs, batch_size=16, lr=1e-4, CosineAnnealingLR
  - ckpt_tag: _exp3_rot  -> checkpoints/cropping_efficientnet_b3_exp3_rot.pt
  - log:      logs/exp3_rot.log

Run:
    python experiments/exp3_rule_of_thirds.py
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import box_iou_numpy, _box_iou_loss

# ---------------------------------------------------------------------------
#  Hyper-parameters
# ---------------------------------------------------------------------------

BACKBONE       = "efficientnet_b3"
EPOCHS         = 10
BATCH_SIZE     = 16
LR             = 1e-4
ROT_LOSS_W     = 0.1     # auxiliary RoT alignment loss weight
PLAYER_LOSS_W  = 0.5     # player-coverage hinge penalty weight
SMOOTH_L1_W    = 0.5     # fraction of SmoothL1 vs (1-IoU) in box loss

LOG_PATH  = ROOT / "logs" / "exp3_rot.log"
CKPT_PATH = CHECKPOINTS_DIR / "cropping_efficientnet_b3_exp3_rot.pt"

_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_UNION_BBOX_CACHE   = ROOT / "data" / "player_bboxes.json"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_EXTRACT_SIZE  = 512

# The 4 RoT intersections in (x, y) order
_ROT_INTERSECTIONS = [
    (1 / 3, 1 / 3),
    (2 / 3, 1 / 3),
    (1 / 3, 2 / 3),
    (2 / 3, 2 / 3),
]


# ---------------------------------------------------------------------------
#  RoT feature computation (12-dim)
# ---------------------------------------------------------------------------

def compute_rot_features(primary_bbox: torch.Tensor) -> torch.Tensor:
    """
    primary_bbox: [B, 4] -- (x1, y1, x2, y2) normalized to [0, 1].

    Returns [B, 12]:
      [0-3]   Euclidean distance from player center to each of 4 RoT
              intersections: (1/3,1/3), (2/3,1/3), (1/3,2/3), (2/3,2/3).
      [4-6]   One-hot: vertical zone of player center
                  left (<1/3), center (1/3..2/3), right (>2/3)
      [7-9]   One-hot: horizontal zone of player center
                  top (<1/3), middle (1/3..2/3), bottom (>2/3)
      [10]    Player width  (x2 - x1)
      [11]    Player height (y2 - y1)
    """
    x1, y1, x2, y2 = primary_bbox.unbind(1)   # each [B]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    pw = (x2 - x1).clamp(min=0.0)
    ph = (y2 - y1).clamp(min=0.0)

    # 4 distances to RoT intersections
    r1, r2 = 1.0 / 3.0, 2.0 / 3.0
    dist_features = []
    for (ix, iy) in _ROT_INTERSECTIONS:
        d = ((cx - ix) ** 2 + (cy - iy) ** 2).sqrt()
        dist_features.append(d)
    dist_t = torch.stack(dist_features, dim=1)   # [B, 4]

    # Vertical zone one-hot (left, center, right)
    zone_left   = (cx < r1).float()
    zone_right  = (cx > r2).float()
    zone_center_x = (1.0 - zone_left - zone_right).clamp(min=0.0)
    v_onehot = torch.stack([zone_left, zone_center_x, zone_right], dim=1)  # [B, 3]

    # Horizontal zone one-hot (top, middle, bottom)
    zone_top    = (cy < r1).float()
    zone_bottom = (cy > r2).float()
    zone_mid_y  = (1.0 - zone_top - zone_bottom).clamp(min=0.0)
    h_onehot = torch.stack([zone_top, zone_mid_y, zone_bottom], dim=1)  # [B, 3]

    # Width and height
    size_t = torch.stack([pw, ph], dim=1)         # [B, 2]

    return torch.cat([dist_t, v_onehot, h_onehot, size_t], dim=1)  # [B, 12]


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

def build_rot_model(backbone: str = BACKBONE, pretrained: bool = True) -> nn.Module:
    backbone_model = timm.create_model(
        backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    in_features    = backbone_model.num_features
    player_emb_dim = 48
    head_in        = in_features + player_emb_dim

    class RoT12CropRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_model
            # 12-dim RoT features from primary bbox
            self.player_encoder = nn.Sequential(
                nn.Linear(12, player_emb_dim),
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
                    primary_bbox: torch.Tensor) -> torch.Tensor:
            feats    = self.backbone(x)
            rot_feat = compute_rot_features(primary_bbox)          # [B, 12]
            feats    = torch.cat([feats, self.player_encoder(rot_feat)], dim=1)
            return self.box_head(feats)

        def set_grad_checkpointing(self, enable: bool = True) -> None:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=enable)

    return RoT12CropRegressor()


# ---------------------------------------------------------------------------
#  RoT auxiliary loss
# ---------------------------------------------------------------------------

def rot_auxiliary_loss(box_pred: torch.Tensor,
                       primary_bbox: torch.Tensor) -> torch.Tensor:
    """
    Rewards placing the primary player center near a RoT intersection
    *inside the predicted crop*.

    For each sample with a detected player (primary_bbox != zeros):
      1. Convert player center to crop-local coordinates.
      2. Penalise distance of the player center from the nearest RoT
         intersection in each axis independently:
             rot_x = min((rel_cx - 1/3)^2, (rel_cx - 2/3)^2)
             rot_y = min((rel_cy - 1/3)^2, (rel_cy - 2/3)^2)
             rot_loss_sample = rot_x + rot_y

    Returns a scalar loss (0.0 graph-connected tensor if no player detected).
    """
    has_player = primary_bbox.sum(dim=1) > 0.0
    if not has_player.any():
        return box_pred.sum() * 0.0

    pb = primary_bbox[has_player]
    pp = box_pred[has_player]

    px1, py1, px2, py2 = pb.unbind(1)
    cx1, cy1, cx2, cy2 = pp.unbind(1)

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    cw  = (cx2 - cx1).clamp(min=1e-6)
    ch  = (cy2 - cy1).clamp(min=1e-6)

    rel_cx = ((pcx - cx1) / cw).clamp(0.0, 1.0)
    rel_cy = ((pcy - cy1) / ch).clamp(0.0, 1.0)

    r1, r2 = 1.0 / 3.0, 2.0 / 3.0
    rot_x = torch.minimum((rel_cx - r1) ** 2, (rel_cx - r2) ** 2)
    rot_y = torch.minimum((rel_cy - r1) ** 2, (rel_cy - r2) ** 2)

    return (rot_x + rot_y).mean()


# ---------------------------------------------------------------------------
#  Combined loss
# ---------------------------------------------------------------------------

class RoT12CropLoss(nn.Module):
    def __init__(self, alpha: float = SMOOTH_L1_W,
                 player_weight: float = PLAYER_LOSS_W,
                 rot_weight: float = ROT_LOSS_W):
        super().__init__()
        self.alpha         = alpha
        self.player_weight = player_weight
        self.rot_weight    = rot_weight

    def forward(self, box_pred: torch.Tensor,
                box_target: torch.Tensor,
                primary_bbox: torch.Tensor):
        sl1      = nn.functional.smooth_l1_loss(box_pred, box_target)
        iou      = _box_iou_loss(box_pred, box_target)
        box_loss = self.alpha * sl1 + (1.0 - self.alpha) * iou

        # Player-coverage hinge penalty
        player_loss = torch.zeros(1, device=box_pred.device)
        has_player  = primary_bbox.sum(dim=1) > 0.0
        if self.player_weight > 0 and has_player.any():
            px1, py1, px2, py2 = primary_bbox.unbind(1)
            cx1, cy1, cx2, cy2 = box_pred.unbind(1)
            clip_per = (
                (cx1 - px1).clamp(min=0.0) +
                (cy1 - py1).clamp(min=0.0) +
                (px2 - cx2).clamp(min=0.0) +
                (py2 - cy2).clamp(min=0.0)
            )
            n_valid     = has_player.float().sum().clamp(min=1.0)
            player_loss = (clip_per * has_player.float()).sum() / n_valid

        # RoT auxiliary loss
        rot_loss = rot_auxiliary_loss(box_pred, primary_bbox)

        total = (box_loss
                 + self.player_weight * player_loss
                 + self.rot_weight * rot_loss)
        return total, box_loss.detach(), rot_loss.detach()


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class RoT12Dataset(Dataset):
    def __init__(self, records, transform, hflip=False,
                 primary_cache=None):
        self.records       = records
        self.transform     = transform
        self.hflip         = hflip
        self.primary_cache = primary_cache or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=_EXTRACT_SIZE)
        box = list(r["box"])

        pb = self.primary_cache.get(r["raw"])
        primary_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]

        if self.hflip and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x1, y1, x2, y2 = box
            box = [1.0 - x2, y1, 1.0 - x1, y2]
            if pb is not None:
                bx1, by1, bx2, by2 = primary_bbox
                primary_bbox = [1.0 - bx2, by1, 1.0 - bx1, by2]

        return (
            self.transform(img),
            torch.tensor(box,          dtype=torch.float32),
            torch.tensor(primary_bbox, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for imgs, boxes, primary_bboxes in loader:
            box_pred = model(imgs.to(device), primary_bboxes.to(device))
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


# ---------------------------------------------------------------------------
#  GT RoT analysis
# ---------------------------------------------------------------------------

def _player_in_crop_local(primary_bbox, crop_box):
    """Player center in crop-local [0,1] coords. Returns None if degenerate."""
    px1, py1, px2, py2 = primary_bbox
    cx1, cy1, cx2, cy2 = crop_box
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None
    if px1 == py1 == px2 == py2 == 0.0:
        return None
    return (((px1 + px2) / 2 - cx1) / cw,
            ((py1 + py2) / 2 - cy1) / ch)


def analyze_rot_adherence(records, primary_cache):
    """
    For all GT records with a detected primary player, compute where the player
    center falls within the GT crop in crop-local coordinates and report what
    fraction are near a RoT intersection.
    """
    local_xs, local_ys = [], []
    skipped = 0
    for r in records:
        pb = primary_cache.get(r["raw"])
        if pb is None:
            skipped += 1
            continue
        result = _player_in_crop_local(pb, r["box"])
        if result is None:
            skipped += 1
            continue
        local_xs.append(result[0])
        local_ys.append(result[1])

    n = len(local_xs)
    if n == 0:
        return {"error": "no samples with GT crop and primary bbox", "skipped": skipped}

    lx = np.array(local_xs)
    ly = np.array(local_ys)

    r1, r2 = 1 / 3, 2 / 3
    dx = np.minimum(np.abs(lx - r1), np.abs(lx - r2))
    dy = np.minimum(np.abs(ly - r1), np.abs(ly - r2))

    # Within 0.10 of a RoT line
    frac_x_near  = float((dx < 0.10).mean())
    frac_y_near  = float((dy < 0.10).mean())
    frac_xy_near = float(((dx < 0.10) & (dy < 0.10)).mean())

    # Distances to all 4 intersections
    best_dist = np.array([
        min(np.sqrt((x - ix) ** 2 + (y - iy) ** 2)
            for ix, iy in _ROT_INTERSECTIONS)
        for x, y in zip(lx, ly)
    ])

    return {
        "n":                    n,
        "skipped":              skipped,
        "mean_local_cx":        float(lx.mean()),
        "std_local_cx":         float(lx.std()),
        "mean_local_cy":        float(ly.mean()),
        "std_local_cy":         float(ly.std()),
        "frac_x_near_rot":      frac_x_near,
        "frac_y_near_rot":      frac_y_near,
        "frac_near_intersection": frac_xy_near,
        "mean_best_dist":       float(best_dist.mean()),
        "median_best_dist":     float(np.median(best_dist)),
        "frac_within_0.10":     float((best_dist < 0.10).mean()),
        "frac_within_0.15":     float((best_dist < 0.15).mean()),
        "frac_within_0.20":     float((best_dist < 0.20).mean()),
    }


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_PATH), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Experiment 3: Rule-of-Thirds (12-dim features + auxiliary loss)")
    log.info(f"  backbone={BACKBONE}  epochs={EPOCHS}  bs={BATCH_SIZE}  lr={LR}")
    log.info(f"  rot_loss_weight={ROT_LOSS_W}  player_loss_weight={PLAYER_LOSS_W}")
    log.info(f"  player_encoder: Linear(12 -> 48, ReLU)")
    log.info(f"  features: 4 RoT-intersection dists + 3 v-zone + 3 h-zone + w + h")
    log.info("=" * 70)

    # -- load GT data --------------------------------------------------------
    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    log.info(f"GT: train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    # -- bbox caches ---------------------------------------------------------
    primary_cache = {}
    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            primary_cache = json.load(fh)
        n_covered = sum(1 for r in all_records if primary_cache.get(r["raw"]) is not None)
        log.info(f"Primary bbox cache: {len(primary_cache):,} entries  "
                 f"({n_covered}/{len(all_records)} GT raws covered)")
    else:
        # Fallback: try union cache
        if _UNION_BBOX_CACHE.exists():
            with open(_UNION_BBOX_CACHE) as fh:
                primary_cache = json.load(fh)
            log.warning("Primary bbox cache not found; falling back to union bbox cache")
        else:
            log.warning("No player bbox caches found -- all conditioning will be zero")

    # -- GT RoT analysis (pre-training) -------------------------------------
    log.info("")
    log.info("-" * 60)
    log.info("GT RULE-OF-THIRDS ANALYSIS  (all splits)")
    analysis = analyze_rot_adherence(all_records, primary_cache)
    if "error" in analysis:
        log.warning(f"  Analysis failed: {analysis['error']}")
    else:
        log.info(f"  Samples with primary bbox: {analysis['n']:,}  "
                 f"(skipped {analysis['skipped']:,})")
        log.info(f"  Player center in crop -- cx: {analysis['mean_local_cx']:.3f} "
                 f"(+/-{analysis['std_local_cx']:.3f})  "
                 f"cy: {analysis['mean_local_cy']:.3f} "
                 f"(+/-{analysis['std_local_cy']:.3f})")
        log.info(f"  Fraction cx near a RoT vertical  (+-0.10): "
                 f"{analysis['frac_x_near_rot']:.1%}")
        log.info(f"  Fraction cy near a RoT horizontal(+-0.10): "
                 f"{analysis['frac_y_near_rot']:.1%}")
        log.info(f"  Fraction near an intersection (both axes) : "
                 f"{analysis['frac_near_intersection']:.1%}")
        log.info(f"  Mean Euclidean dist to nearest intersection: "
                 f"{analysis['mean_best_dist']:.3f}  "
                 f"(median {analysis['median_best_dist']:.3f})")
        log.info(f"  Within 0.10 of nearest intersection: "
                 f"{analysis['frac_within_0.10']:.1%}")
        log.info(f"  Within 0.15 of nearest intersection: "
                 f"{analysis['frac_within_0.15']:.1%}")
        log.info(f"  Within 0.20 of nearest intersection: "
                 f"{analysis['frac_within_0.20']:.1%}")
        if analysis["frac_near_intersection"] > 0.25:
            log.info("  VERDICT: RoT hypothesis SUPPORTED (>25% at intersection)")
        else:
            log.info(f"  VERDICT: RoT hypothesis WEAK (<= 25% at intersection; "
                     f"actual {analysis['frac_near_intersection']:.1%})")
    log.info("-" * 60)
    log.info("")

    # -- device / model ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = build_rot_model(backbone=BACKBONE, pretrained=True).to(device)

    data_cfg   = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean  = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    norm_std   = tuple(data_cfg.get("std",  _IMAGENET_STD))
    log.info(f"Input size: {input_size}x{input_size}")

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
        RoT12Dataset(train_recs, tf_train, hflip=True, primary_cache=primary_cache),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        RoT12Dataset(val_recs, tf_val, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        RoT12Dataset(test_recs, tf_val, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )

    criterion = RoT12CropLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_iou      = -1.0
    best_metrics: dict = {}
    val_iou_per_epoch = []

    log.info(f"Training {BACKBONE} for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss     = 0.0
        total_box_loss = 0.0
        total_rot_loss = 0.0
        n_batches      = 0

        for imgs, boxes, primary_bboxes in tqdm(
                train_loader, desc=f"ep{epoch}/{EPOCHS}", leave=False):
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            primary_bboxes = primary_bboxes.to(device)

            optimizer.zero_grad()
            box_pred = model(imgs, primary_bboxes)
            loss, box_l, rot_l = criterion(box_pred, boxes, primary_bboxes)
            loss.backward()
            optimizer.step()

            total_loss     += loss.item()
            total_box_loss += box_l.item()
            total_rot_loss += rot_l.item()
            n_batches      += 1

        sched.step()

        val_m = evaluate(model, val_loader, device)
        val_iou_per_epoch.append(val_m["mean_iou"])

        avg_loss     = total_loss     / max(n_batches, 1)
        avg_box_loss = total_box_loss / max(n_batches, 1)
        avg_rot_loss = total_rot_loss / max(n_batches, 1)

        log.info(
            f"  ep{epoch:02d}"
            f"  loss={avg_loss:.4f}"
            f"  box_loss={avg_box_loss:.4f}"
            f"  rot_loss={avg_rot_loss:.4f}"
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
                "player_loss_w": PLAYER_LOSS_W,
                "feature_dim":  12,
                "player_emb_dim": 48,
            }, str(CKPT_PATH))
            log.info(f"    [SAVED] best val IoU={best_iou:.4f}")

    # -- test evaluation -----------------------------------------------------
    ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = evaluate(model, test_loader, device)
    log.info("")
    log.info(f"TEST  mean_iou={test_m['mean_iou']:.4f}"
             f"  median={test_m['median_iou']:.4f}"
             f"  >0.7:{test_m['iou_gt70']:.1%}"
             f"  >0.8:{test_m['iou_gt80']:.1%}")

    # -- per-prediction RoT analysis (test set) -----------------------------
    log.info("")
    log.info("-" * 60)
    log.info("POST-TRAINING: Predicted crop RoT analysis (test set)")
    model.eval()
    pred_local_xs, pred_local_ys = [], []
    with torch.no_grad():
        for imgs, boxes, primary_bboxes in DataLoader(
                RoT12Dataset(test_recs, tf_val, primary_cache=primary_cache),
                batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw):
            box_pred       = model(imgs.to(device), primary_bboxes.to(device))
            box_pred_np    = box_pred.cpu().numpy()
            primary_np     = primary_bboxes.numpy()
            for pred_box, pb in zip(box_pred_np, primary_np):
                result = _player_in_crop_local(pb, pred_box)
                if result is not None:
                    pred_local_xs.append(result[0])
                    pred_local_ys.append(result[1])

    if pred_local_xs:
        plx = np.array(pred_local_xs)
        ply = np.array(pred_local_ys)
        r1, r2 = 1 / 3, 2 / 3
        pdx = np.minimum(np.abs(plx - r1), np.abs(plx - r2))
        pdy = np.minimum(np.abs(ply - r1), np.abs(ply - r2))
        pfrac_near = float(((pdx < 0.10) & (pdy < 0.10)).mean())
        best_pdist = np.array([
            min(np.sqrt((x - ix) ** 2 + (y - iy) ** 2)
                for ix, iy in _ROT_INTERSECTIONS)
            for x, y in zip(plx, ply)
        ])
        log.info(f"  Predicted crops analysed: {len(plx):,}")
        log.info(f"  Mean predicted cx in crop: {plx.mean():.3f} (+/-{plx.std():.3f})")
        log.info(f"  Mean predicted cy in crop: {ply.mean():.3f} (+/-{ply.std():.3f})")
        log.info(f"  Fraction near intersection (both+-0.10): {pfrac_near:.1%}")
        log.info(f"  Mean dist to nearest intersection: {best_pdist.mean():.3f}")
        log.info(f"  Baseline (if not trained): random uniform expected ~11.1% at intersection")
        if analysis and "frac_near_intersection" in analysis:
            log.info(f"  GT adherence:        {analysis['frac_near_intersection']:.1%}")
            log.info(f"  Predicted adherence: {pfrac_near:.1%}")
            gain = pfrac_near - analysis["frac_near_intersection"]
            log.info(f"  Delta (pred - GT):   {gain:+.1%}")
    log.info("-" * 60)

    # -- summary -------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info(f"  Best val IoU : {best_iou:.4f}  (epoch {best_metrics.get('epoch')})")
    log.info(f"  Test IoU     : {test_m['mean_iou']:.4f}")
    log.info(f"  Val IoU per epoch:")
    for ep, iou in enumerate(val_iou_per_epoch, 1):
        marker = "  <-- best" if iou == best_iou else ""
        log.info(f"    ep{ep:02d}: {iou:.4f}{marker}")
    log.info(f"  Checkpoint   : {CKPT_PATH}")
    log.info(f"  Log          : {LOG_PATH}")

    if analysis and "frac_near_intersection" in analysis:
        log.info("")
        log.info("RoT ANALYSIS VERDICT:")
        fni = analysis["frac_near_intersection"]
        log.info(f"  GT crop RoT adherence: {fni:.1%} of player centers "
                 f"within 0.10 of an intersection")
        log.info(f"  Mean cx: {analysis['mean_local_cx']:.3f}  "
                 f"Mean cy: {analysis['mean_local_cy']:.3f}")
        if fni > 0.30:
            log.info("  RECOMMENDATION: Strong RoT signal -- keep RoT conditioning.")
        elif fni > 0.20:
            log.info("  RECOMMENDATION: Moderate RoT signal -- conditioning may help "
                     "but verify IoU vs baseline.")
        else:
            log.info("  RECOMMENDATION: Weak RoT signal -- RoT conditioning likely "
                     "adds noise; consider dropping.")
    log.info("=" * 70)

    return {
        "analysis":          analysis,
        "val_iou_per_epoch": val_iou_per_epoch,
        "best_val_iou":      best_iou,
        "test_metrics":      test_m,
    }


if __name__ == "__main__":
    main()
