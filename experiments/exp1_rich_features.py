"""
Experiment 1: Richer player feature conditioning for the crop regressor.

Instead of passing raw [x1, y1, x2, y2] (4 numbers) to the model, we derive
13 analytical features from the union + primary bbox pair, giving the model
explicit compositional cues it would otherwise have to learn implicitly.

Feature vector (13-dim):
  From union bbox [x1,y1,x2,y2]:
    0  cx             – centre-x
    1  cy             – centre-y
    2  w              – width
    3  h              – height
    4  area           – w * h
    5  aspect_ratio   – w / (h + eps)
    6  dx_from_centre – cx - 0.5  (signed; positive = right of frame)
    7  dy_from_centre – cy - 0.5  (positive = below centre)
    8  dist_centre    – sqrt(dx^2 + dy^2)
    9  vertical_third – 0=top third  1=middle  2=bottom  (ordinal float /2)
    10 area_ratio     – area / 1.0  (same as area here; explicit shot-distance proxy)

  From union vs primary comparison (doubles-detection proxy):
    11 area_ratio_u_over_p  – union_area / (primary_area + eps)
                               ≈ 1.0 for singles, > 1 for doubles
    12 has_two_players      – 1.0 if area_ratio_u_over_p > 1.4 else 0.0

Both bbox inputs fall back to [0,0,0,0] when not detected; the first 11 dims
collapse to zeros, 11 → 1.0, 12 → 0.0, which is a learnable "no detection" mode.

Architecture change:
  player_encoder: Linear(13 → 64) + ReLU  (was Linear(4 → 32))
  box_head: backbone_feats + 64 → 256 → 4

All other design choices (backbone, loss, data split) are unchanged from the
production baseline.

Usage:
    python experiments/exp1_rich_features.py

Outputs:
    logs/exp1_rich_features.log
    checkpoints/cropping_angle_efficientnet_b3_exp1.pt
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

# ── constants ──────────────────────────────────────────────────────────────────
BACKBONE           = "efficientnet_b3"
EPOCHS             = 10
BATCH_SIZE         = 16
LR                 = 1e-4
NUM_WORKERS        = 4
PLAYER_EMB_DIM     = 64          # increased from 32 in baseline
RICH_FEAT_DIM      = 13
ALPHA              = 0.5         # SmoothL1 vs IoU blend in loss
PLAYER_WEIGHT      = 0.5         # player-coverage penalty weight
ANGLE_SCALE        = 90.0

_PLAYER_BBOX_CACHE  = ROOT / "data" / "player_bboxes.json"
_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_LOG_FILE           = ROOT / "logs" / "exp1_rich_features.log"
_CKPT_FILE          = CHECKPOINTS_DIR / "cropping_angle_efficientnet_b3_exp1.pt"
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


# ── rich feature extractor ─────────────────────────────────────────────────────
def build_rich_features(union_bbox: torch.Tensor,
                        primary_bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert [B,4] union + [B,4] primary bboxes → [B, RICH_FEAT_DIM] feature tensor.

    All values are normalized to roughly [0, 1] so they are on the same scale as
    the bbox coordinates themselves.  The computation is fully differentiable
    (no branches on per-sample values) but we only *use* it as static conditioning
    (no gradients flow back into bbox inputs).
    """
    eps = 1e-6
    ux1, uy1, ux2, uy2 = union_bbox.unbind(1)   # each [B]
    px1, py1, px2, py2 = primary_bbox.unbind(1)

    uw = (ux2 - ux1).clamp(min=0.0)
    uh = (uy2 - uy1).clamp(min=0.0)
    cx = ux1 + uw / 2.0
    cy = uy1 + uh / 2.0
    u_area = uw * uh

    pw = (px2 - px1).clamp(min=0.0)
    ph = (py2 - py1).clamp(min=0.0)
    p_area = pw * ph

    dx = cx - 0.5
    dy = cy - 0.5
    dist = (dx ** 2 + dy ** 2).sqrt()

    # Vertical third: 0=top, 0.5=middle, 1=bottom  (ordinal, rescaled to [0,1])
    vert_third = (cy / (1.0 / 3.0 + eps)).clamp(max=2.0) / 2.0

    # Doubles proxy: union much bigger than primary → two players visible
    area_ratio = u_area / (p_area + eps)
    has_two    = (area_ratio > 1.4).float()

    # Stack → [B, 13]
    feats = torch.stack([
        cx,                          # 0
        cy,                          # 1
        uw,                          # 2
        uh,                          # 3
        u_area,                      # 4
        uw / (uh + eps),             # 5  aspect ratio
        dx,                          # 6
        dy,                          # 7
        dist,                        # 8
        vert_third,                  # 9
        u_area,                      # 10 area_ratio proxy (same as area; explicit)
        (area_ratio - 1.0).clamp(min=0.0),  # 11 excess area beyond primary
        has_two,                     # 12
    ], dim=1)
    return feats


# ── model ──────────────────────────────────────────────────────────────────────
class EnrichedCropModel(nn.Module):
    """
    EfficientNet-B3 backbone + box head conditioned on a 13-dim analytical
    feature vector derived from player bbox(es).

    Compared to baseline CropRegressor:
      - player_encoder: Linear(13→64)+ReLU  instead of Linear(4→32)+ReLU
      - box_head input dim: backbone_feats + 64  instead of + 32
      - everything else identical
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

        in_features   = self.backbone.num_features
        head_in        = in_features + PLAYER_EMB_DIM

        self.player_encoder = nn.Sequential(
            nn.Linear(RICH_FEAT_DIM, PLAYER_EMB_DIM),
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
                rich_feats: torch.Tensor) -> torch.Tensor:
        vis = self.backbone(x)                        # [B, in_features]
        emb = self.player_encoder(rich_feats)         # [B, 64]
        return self.box_head(torch.cat([vis, emb], dim=1))  # [B, 4]


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
    sl1      = nn.functional.smooth_l1_loss(box_pred, box_target)
    iou      = _box_iou_loss(box_pred, box_target)
    box_loss = ALPHA * sl1 + (1.0 - ALPHA) * iou

    # Player-coverage hinge penalty (same as CropLoss in model.py)
    has_player = primary_bbox.sum(dim=1) > 0.0
    if has_player.any():
        pb  = primary_bbox[has_player]
        pp  = box_pred[has_player]
        px1, py1, px2, py2 = pb.unbind(1)
        cx1, cy1, cx2, cy2 = pp.unbind(1)
        clip = (
            (cx1 - px1).clamp(min=0.0) +
            (cy1 - py1).clamp(min=0.0) +
            (px2 - cx2).clamp(min=0.0) +
            (py2 - cy2).clamp(min=0.0)
        )
        box_loss = box_loss + PLAYER_WEIGHT * clip.mean()

    return box_loss


# ── dataset ────────────────────────────────────────────────────────────────────
class RichCropDataset(Dataset):
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

        img_t         = self.transform(img)
        box_t         = torch.tensor(box,         dtype=torch.float32)
        union_bbox_t  = torch.tensor(union_bbox,  dtype=torch.float32)
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
        imgs         = imgs.to(device)
        union_bboxes = union_bboxes.to(device)
        primary_bboxes = primary_bboxes.to(device)
        rich = build_rich_features(union_bboxes, primary_bboxes)
        out  = model(imgs, rich)
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


# ── main training loop ─────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 70)
    log.info("Experiment 1: Rich player feature conditioning")
    log.info(f"  backbone={BACKBONE}  epochs={EPOCHS}  bs={BATCH_SIZE}  lr={LR}")
    log.info(f"  feature_dim={RICH_FEAT_DIM}  emb_dim={PLAYER_EMB_DIM}")
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
        log.info(f"  Union bbox cache: {len(union_cache):,} entries  "
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
        log.warning(f"  Primary bbox cache not found; using union as fallback")
        primary_cache = union_cache

    # ── transforms ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  device={device}")

    model    = EnrichedCropModel(backbone=BACKBONE, pretrained=True).to(device)
    data_cfg = timm.data.resolve_model_data_config(model.backbone)
    inp_sz   = data_cfg.get("input_size", (3, 224, 224))[1]
    mean_    = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    std_     = tuple(data_cfg.get("std",  _IMAGENET_STD))

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
        RichCropDataset(train_recs, tf_train, hflip=True,
                        union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        RichCropDataset(val_recs, tf_val,
                        union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        RichCropDataset(test_recs, tf_val,
                        union_cache=union_cache, primary_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    _CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    best_iou     = -1.0
    best_metrics: dict = {}

    log.info(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val mIoU':<10} "
             f"{'Val med':<10} {'>0.7':<8} {'>0.8':<8}")
    log.info("-" * 60)

    iou_history: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, boxes, union_bboxes, primary_bboxes in train_loader:
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            union_bboxes   = union_bboxes.to(device)
            primary_bboxes = primary_bboxes.to(device)

            rich = build_rich_features(union_bboxes, primary_bboxes)
            box_pred = model(imgs, rich)
            loss = crop_loss(box_pred, boxes, primary_bboxes)

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
                "epoch":        epoch,
                "backbone":     BACKBONE,
                "feat_dim":     RICH_FEAT_DIM,
                "emb_dim":      PLAYER_EMB_DIM,
                "model_state":  model.state_dict(),
                "metrics":      val_m,
                "input_size":   inp_sz,
                "norm_mean":    mean_,
                "norm_std":     std_,
            }, _CKPT_FILE)
            log.info(f"    [BEST] Saved  val_iou={best_iou:.4f}")

    # ── test evaluation on best checkpoint ────────────────────────────────────
    ck = torch.load(str(_CKPT_FILE), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = evaluate(model, test_loader, device)
    log.info("")
    log.info("=" * 60)
    log.info(f"TEST RESULTS  (best val ep{best_metrics['epoch']})")
    log.info(f"  mean_iou  = {test_m['mean_iou']:.4f}")
    log.info(f"  median    = {test_m['median_iou']:.4f}")
    log.info(f"  >0.70     = {test_m['iou_gt70']:.1%}")
    log.info(f"  >0.80     = {test_m['iou_gt80']:.1%}")
    log.info("=" * 60)

    # ── summary table ─────────────────────────────────────────────────────────
    log.info("")
    log.info("Per-epoch val IoU summary:")
    log.info(f"  {'Epoch':<6} {'Val mIoU'}")
    for i, v in enumerate(iou_history, 1):
        marker = "  <-- best" if abs(v - best_iou) < 1e-7 else ""
        log.info(f"  {i:<6} {v:.4f}{marker}")

    log.info("")
    log.info("Baseline reference (25 ep, efficientnet_b3, box+angle+player_bbox):")
    log.info("  val mean_iou = 0.819")
    log.info("")
    log.info(f"Experiment best val mean_iou = {best_iou:.4f}  "
             f"(ep {best_metrics['epoch']}/{EPOCHS})")
    delta = best_iou - 0.819
    sign  = "+" if delta >= 0 else ""
    log.info(f"Delta vs baseline: {sign}{delta:+.4f}")
    log.info("")
    log.info("Rich feature breakdown (what each dim encodes):")
    log.info("  0,1  cx,cy          – player centre position (composition guidance)")
    log.info("  2,3  w,h            – player size (shot distance proxy)")
    log.info("  4    area           – bounding-box area")
    log.info("  5    aspect_ratio   – portrait vs landscape body pose")
    log.info("  6,7  dx,dy          – signed distance from frame centre")
    log.info("  8    dist_centre    – Euclidean distance from centre (magnitude)")
    log.info("  9    vert_third     – vertical zone (top/mid/bottom of frame)")
    log.info("  10   area (repeat)  – explicit shot-distance signal")
    log.info("  11   excess_area    – union−primary area gap (doubles proxy)")
    log.info("  12   has_two_players– binary doubles flag")

    best_metrics["test_metrics"] = test_m
    return best_metrics


if __name__ == "__main__":
    main()
