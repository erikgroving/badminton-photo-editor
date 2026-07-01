"""
Experiment: Explicit two-player / doubles conditioning for the crop regressor.

Hypothesis: Jay Ma's key composition decision is singles (tight portrait crop)
vs doubles (wider landscape crop). The current model only sees the union bbox.
By explicitly encoding the PRIMARY player's bbox AND the union bbox — plus derived
doubles-detection features — the model can explicitly distinguish shot types.

Feature vector (10-dim):
  [0..3]  primary_x1, primary_y1, primary_x2, primary_y2
  [4..7]  union_x1,   union_y1,   union_x2,   union_y2
  [8]     secondary_presence  – area(union)/area(primary); clipped if > 1.3 → doubles
  [9]     inter_player_gap    – (union_w - primary_w) / union_w; how spread the players are

Architecture change:
  player_encoder: Linear(10 → 64) + ReLU  (baseline used Linear(4 → 32) on union only)
  box_head:       backbone_feats(1536) + 64 → 256 → ReLU → Dropout(0.3) → 4 → Sigmoid

Checkpoint tag: _exp2_twoplayer  (distinct from exp2 / exp2_rot)
Log:            logs/exp2_two_player.log
Checkpoint:     checkpoints/cropping_efficientnet_b3_exp2_twoplayer.pt

Usage:
    python experiments/exp2_two_player.py
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

# ── project root on path ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar

# ── hyper-parameters ───────────────────────────────────────────────────────────
BACKBONE        = "efficientnet_b3"
EPOCHS          = 10
BATCH_SIZE      = 16
LR              = 1e-4
NUM_WORKERS     = 4
FEAT_DIM        = 10      # [primary(4) + union(4) + secondary_presence + inter_gap]
PLAYER_EMB_DIM  = 64
ALPHA           = 0.5     # SmoothL1 vs IoU blend
PLAYER_WEIGHT   = 0.5     # player-coverage hinge penalty weight
CKPT_TAG        = "_exp2_twoplayer"
BASELINE_IOU    = 0.819   # efficientnet_b3, 25ep, test IoU

# ── paths ──────────────────────────────────────────────────────────────────────
_PLAYER_BBOX_CACHE  = ROOT / "data" / "player_bboxes.json"          # union bbox
_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"  # largest player
_LOG_FILE           = ROOT / "logs" / "exp2_two_player.log"
_CKPT_FILE          = CHECKPOINTS_DIR / f"cropping_efficientnet_b3{CKPT_TAG}.pt"
_IMAGENET_MEAN      = (0.485, 0.456, 0.406)
_IMAGENET_STD       = (0.229, 0.224, 0.225)
_EXTRACT_SIZE       = 512

# ── logging ─────────────────────────────────────────────────────────────────────
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── feature engineering ────────────────────────────────────────────────────────
def build_two_player_features(union_bbox: torch.Tensor,
                              primary_bbox: torch.Tensor) -> torch.Tensor:
    """
    [B, 4] union + [B, 4] primary  →  [B, FEAT_DIM=10] feature tensor.

    Dims:
      0..3  primary_x1, primary_y1, primary_x2, primary_y2
      4..7  union_x1,   union_y1,   union_x2,   union_y2
      8     secondary_presence  = area(union) / (area(primary) + eps)
                                  clamped to [1, 2]; > 1.3 ≈ doubles shot
      9     inter_player_gap    = (union_w - primary_w) / (union_w + eps)
                                  0 = single player; > 0 = players spread horizontally
    """
    eps = 1e-6

    ux1, uy1, ux2, uy2 = union_bbox.unbind(1)     # [B] each
    px1, py1, px2, py2 = primary_bbox.unbind(1)

    uw = (ux2 - ux1).clamp(min=0.0)
    uh = (uy2 - uy1).clamp(min=0.0)
    pw = (px2 - px1).clamp(min=0.0)
    ph = (py2 - py1).clamp(min=0.0)

    u_area = uw * uh
    p_area = pw * ph

    # secondary_presence: ratio of union area to primary area
    # Singles  → ~1.0  (union ≈ primary)
    # Doubles  → > 1.3 (union covers two people)
    secondary_presence = (u_area / (p_area + eps)).clamp(1.0, 2.0)

    # inter_player_gap: fraction of union width not explained by primary
    inter_player_gap = ((uw - pw) / (uw + eps)).clamp(0.0, 1.0)

    # Stack → [B, 10]
    feats = torch.stack([
        px1,                 # 0  primary left edge
        py1,                 # 1  primary top edge
        px2,                 # 2  primary right edge
        py2,                 # 3  primary bottom edge
        ux1,                 # 4  union left edge
        uy1,                 # 5  union top edge
        ux2,                 # 6  union right edge
        uy2,                 # 7  union bottom edge
        secondary_presence,  # 8  doubles indicator (area ratio, clipped to [1,2])
        inter_player_gap,    # 9  horizontal spread between players
    ], dim=1)
    return feats  # [B, 10]


# ── model ──────────────────────────────────────────────────────────────────────
class TwoPlayerCropModel(nn.Module):
    """
    EfficientNet-B3 backbone + box head conditioned on a 10-dim two-player
    feature vector.

    vs. baseline CropRegressor:
      - player_encoder: Linear(10→64)+ReLU  instead of Linear(4→32)+ReLU
      - box_head input:  backbone_feats + 64  (same total, just wider emb)
      - no angle head (keeps experiment focused on bbox-only comparison)
    """
    def __init__(self, backbone: str = BACKBONE, pretrained: bool = True):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained,
                num_classes=0, global_pool="avg",
            )
        except RuntimeError:
            self.backbone = timm.create_model(
                backbone, pretrained=False,
                num_classes=0, global_pool="avg",
            )
            log.warning("  timm pretrained download failed; using random init for backbone")

        in_features = self.backbone.num_features
        head_in     = in_features + PLAYER_EMB_DIM

        # Encode the 10-dim two-player feature vector
        self.player_encoder = nn.Sequential(
            nn.Linear(FEAT_DIM, PLAYER_EMB_DIM),
            nn.ReLU(),
        )
        self.box_head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self,
                x: torch.Tensor,
                two_player_feats: torch.Tensor) -> torch.Tensor:
        vis = self.backbone(x)                                   # [B, in_features]
        emb = self.player_encoder(two_player_feats)              # [B, 64]
        return self.box_head(torch.cat([vis, emb], dim=1))       # [B, 4]


# ── loss ───────────────────────────────────────────────────────────────────────
def _box_iou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ix1   = torch.max(pred[:, 0], gt[:, 0])
    iy1   = torch.max(pred[:, 1], gt[:, 1])
    ix2   = torch.min(pred[:, 2], gt[:, 2])
    iy2   = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pw    = (pred[:, 2] - pred[:, 0]).clamp(0)
    ph    = (pred[:, 3] - pred[:, 1]).clamp(0)
    gw    = (gt[:, 2]   - gt[:, 0]).clamp(0)
    gh    = (gt[:, 3]   - gt[:, 1]).clamp(0)
    union = pw * ph + gw * gh - inter
    return (1.0 - inter / (union + 1e-6)).mean()


def crop_loss(box_pred: torch.Tensor, box_target: torch.Tensor,
              primary_bbox: torch.Tensor) -> torch.Tensor:
    """
    Combined SmoothL1 + (1-IoU) loss with player-coverage hinge penalty on
    the primary player bbox (ensures crop doesn't clip the subject).
    """
    sl1      = nn.functional.smooth_l1_loss(box_pred, box_target)
    iou      = _box_iou_loss(box_pred, box_target)
    box_loss = ALPHA * sl1 + (1.0 - ALPHA) * iou

    # Player-coverage hinge: penalise any edge of pred that clips primary player
    has_player = primary_bbox.sum(dim=1) > 0.0   # mask out zero-detection samples
    if has_player.any():
        pb  = primary_bbox[has_player]
        pp  = box_pred[has_player]
        px1, py1, px2, py2 = pb.unbind(1)
        cx1, cy1, cx2, cy2 = pp.unbind(1)
        clip = (
            (cx1 - px1).clamp(min=0.0) +  # crop left cuts into player left edge
            (cy1 - py1).clamp(min=0.0) +  # crop top cuts into player head
            (px2 - cx2).clamp(min=0.0) +  # crop right cuts into player right edge
            (py2 - cy2).clamp(min=0.0)    # crop bottom cuts into player feet
        )
        box_loss = box_loss + PLAYER_WEIGHT * clip.mean()

    return box_loss


# ── dataset ────────────────────────────────────────────────────────────────────
class TwoPlayerDataset(Dataset):
    def __init__(self, records: list[dict], transform,
                 hflip: bool = False,
                 union_cache: dict | None = None,
                 primary_cache: dict | None = None):
        self.records       = records
        self.transform     = transform
        self.hflip         = hflip
        self.union_cache   = union_cache   or {}
        self.primary_cache = primary_cache or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
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

        img_t          = self.transform(img)
        box_t          = torch.tensor(box,          dtype=torch.float32)
        union_bbox_t   = torch.tensor(union_bbox,   dtype=torch.float32)
        primary_bbox_t = torch.tensor(primary_bbox, dtype=torch.float32)
        return img_t, box_t, union_bbox_t, primary_bbox_t


# ── evaluation ─────────────────────────────────────────────────────────────────
def box_iou_numpy(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    ix1   = np.maximum(pred[:, 0], gt[:, 0])
    iy1   = np.maximum(pred[:, 1], gt[:, 1])
    ix2   = np.minimum(pred[:, 2], gt[:, 2])
    iy2   = np.minimum(pred[:, 3], gt[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    pw    = np.maximum(0, pred[:, 2] - pred[:, 0])
    ph    = np.maximum(0, pred[:, 3] - pred[:, 1])
    gw    = np.maximum(0, gt[:, 2]   - gt[:, 0])
    gh    = np.maximum(0, gt[:, 3]   - gt[:, 1])
    union = pw * ph + gw * gh - inter
    return inter / np.maximum(union, 1e-6)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    all_pred, all_gt = [], []
    for imgs, boxes, union_bboxes, primary_bboxes in loader:
        imgs           = imgs.to(device)
        union_bboxes   = union_bboxes.to(device)
        primary_bboxes = primary_bboxes.to(device)
        two_player_f   = build_two_player_features(union_bboxes, primary_bboxes)
        out            = model(imgs, two_player_f)
        all_pred.append(out.cpu().numpy())
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


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 70)
    log.info("Experiment: Explicit two-player / doubles conditioning")
    log.info(f"  backbone={BACKBONE}  epochs={EPOCHS}  bs={BATCH_SIZE}  lr={LR}")
    log.info(f"  feat_dim={FEAT_DIM}  emb_dim={PLAYER_EMB_DIM}  ckpt_tag={CKPT_TAG}")
    log.info("")
    log.info("  Feature vector (10-dim):")
    log.info("    [0..3]  primary_x1, primary_y1, primary_x2, primary_y2")
    log.info("    [4..7]  union_x1,   union_y1,   union_x2,   union_y2")
    log.info("    [8]     secondary_presence  = area(union)/area(primary), clip [1,2]")
    log.info("    [9]     inter_player_gap    = (union_w - primary_w) / union_w")
    log.info("=" * 70)

    # ── ground truth ──────────────────────────────────────────────────────────
    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)

    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    log.info(f"  train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    # ── bbox caches ───────────────────────────────────────────────────────────
    union_cache: dict = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            union_cache = json.load(fh)
        n_cov = sum(1 for r in all_records if union_cache.get(r["raw"]) is not None)
        log.info(f"  Union bbox cache:   {len(union_cache):,} entries  "
                 f"({n_cov}/{len(all_records)} GT covered)")
    else:
        log.warning(f"  Union bbox cache not found: {_PLAYER_BBOX_CACHE}")

    primary_cache: dict = {}
    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            primary_cache = json.load(fh)
        n_pri = sum(1 for r in all_records if primary_cache.get(r["raw"]) is not None)
        log.info(f"  Primary bbox cache: {len(primary_cache):,} entries  "
                 f"({n_pri}/{len(all_records)} GT covered)")
    else:
        log.warning("  Primary bbox cache not found; falling back to union bbox")
        primary_cache = union_cache

    # ── shot-type analysis: singles vs doubles in training data ───────────────
    singles, doubles, no_detect = 0, 0, 0
    for r in all_records:
        ub = union_cache.get(r["raw"])
        pb = primary_cache.get(r["raw"])
        if ub is None or pb is None:
            no_detect += 1
            continue
        ux1, uy1, ux2, uy2 = ub
        px1, py1, px2, py2 = pb
        u_area = max(0, ux2 - ux1) * max(0, uy2 - uy1)
        p_area = max(0, px2 - px1) * max(0, py2 - py1)
        ratio  = u_area / (p_area + 1e-6)
        if ratio > 1.3:
            doubles += 1
        else:
            singles += 1
    total_det = singles + doubles
    log.info(f"\n  Shot-type breakdown (area_ratio threshold = 1.3):")
    log.info(f"    Singles (ratio <= 1.3): {singles:,}  ({singles/max(total_det,1):.1%})")
    log.info(f"    Doubles (ratio >  1.3): {doubles:,}  ({doubles/max(total_det,1):.1%})")
    log.info(f"    No detection:           {no_detect:,}")

    # ── model + transforms ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n  device={device}")

    model    = TwoPlayerCropModel(backbone=BACKBONE, pretrained=True).to(device)
    data_cfg = timm.data.resolve_model_data_config(model.backbone)
    inp_sz   = data_cfg.get("input_size", (3, 224, 224))[1]
    mean_    = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    std_     = tuple(data_cfg.get("std",  _IMAGENET_STD))
    log.info(f"  input_size={inp_sz}  backbone_feats={model.backbone.num_features}")

    tf_val = transforms.Compose([
        transforms.Resize((inp_sz, inp_sz)),
        transforms.ToTensor(),
        transforms.Normalize(list(mean_), list(std_)),
    ])
    tf_train = transforms.Compose([
        transforms.Resize((inp_sz, inp_sz)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(list(mean_), list(std_)),
    ])

    nw = min(NUM_WORKERS, BATCH_SIZE)
    train_loader = DataLoader(
        TwoPlayerDataset(train_recs, tf_train, hflip=True,
                         union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        TwoPlayerDataset(val_recs, tf_val,
                         union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        TwoPlayerDataset(test_recs, tf_val,
                         union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    _CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    best_iou     = -1.0
    best_metrics: dict = {}
    iou_history:  list[float] = []

    log.info(f"\n{'─'*70}")
    log.info(f"  {'Ep':<5} {'Train Loss':<12} {'Val mIoU':<10} "
             f"{'Val med':<10} {'>0.7':<8} {'>0.8':<8}")
    log.info(f"{'─'*70}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, boxes, union_bboxes, primary_bboxes in train_loader:
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            union_bboxes   = union_bboxes.to(device)
            primary_bboxes = primary_bboxes.to(device)

            two_player_f = build_two_player_features(union_bboxes, primary_bboxes)
            box_pred     = model(imgs, two_player_f)
            loss         = crop_loss(box_pred, boxes, primary_bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_m = evaluate(model, val_loader, device)
        iou_history.append(val_m["mean_iou"])

        log.info(
            f"  ep{epoch:02d}  "
            f"loss={avg_loss:.4f}  "
            f"[val] mean_iou={val_m['mean_iou']:.4f}  "
            f"med={val_m['median_iou']:.4f}  "
            f">0.7:{val_m['iou_gt70']:.1%}  "
            f">0.8:{val_m['iou_gt80']:.1%}"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m, epoch=epoch, backbone=BACKBONE)
            torch.save({
                "epoch":       epoch,
                "backbone":    BACKBONE,
                "feat_dim":    FEAT_DIM,
                "emb_dim":     PLAYER_EMB_DIM,
                "model_state": model.state_dict(),
                "metrics":     val_m,
                "input_size":  inp_sz,
                "norm_mean":   mean_,
                "norm_std":    std_,
                "ckpt_tag":    CKPT_TAG,
            }, _CKPT_FILE)
            log.info(f"    [BEST] Saved  val_iou={best_iou:.4f}")

    # ── test evaluation on best checkpoint ────────────────────────────────────
    ck = torch.load(str(_CKPT_FILE), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = evaluate(model, test_loader, device)

    log.info("")
    log.info("=" * 70)
    log.info(f"TEST RESULTS  (best val ep{best_metrics['epoch']})")
    log.info(f"  mean_iou  = {test_m['mean_iou']:.4f}")
    log.info(f"  median    = {test_m['median_iou']:.4f}")
    log.info(f"  >0.70     = {test_m['iou_gt70']:.1%}")
    log.info(f"  >0.80     = {test_m['iou_gt80']:.1%}")
    log.info("=" * 70)

    # ── summary table ─────────────────────────────────────────────────────────
    log.info("")
    log.info("Per-epoch val IoU:")
    log.info(f"  {'Ep':<5}  {'Val mIoU'}")
    for i, v in enumerate(iou_history, 1):
        marker = "  <-- best" if abs(v - best_iou) < 1e-7 else ""
        log.info(f"  {i:<5}  {v:.4f}{marker}")

    log.info("")
    log.info("Comparison vs baseline:")
    log.info(f"  Baseline (efficientnet_b3, 25ep, union bbox only): test IoU = {BASELINE_IOU:.3f}")
    delta = test_m["mean_iou"] - BASELINE_IOU
    sign  = "+" if delta >= 0 else ""
    log.info(f"  This exp  (efficientnet_b3, {EPOCHS}ep, two-player features): "
             f"test IoU = {test_m['mean_iou']:.4f}  ({sign}{delta:+.4f} vs baseline)")
    log.info("")
    if delta > 0.005:
        log.info("  RESULT: Two-player conditioning HELPS.  "
                 "Recommend training to 25+ ep with this feature set.")
    elif delta > -0.005:
        log.info("  RESULT: Two-player conditioning is NEUTRAL vs baseline at 10ep.  "
                 "May benefit from more epochs or combining with exp1 features.")
    else:
        log.info("  RESULT: Two-player conditioning HURTS at 10ep.  "
                 "The 10-dim feature set may be redundant given what the backbone already sees.")

    best_metrics["test_metrics"] = test_m
    return best_metrics


if __name__ == "__main__":
    main()
