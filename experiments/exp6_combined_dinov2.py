"""
Experiment 6: DINOv2 ViT-B with combined conditioning signals.

Combines the three conditioning signals proven useful by exp1-5 into a single
model built on the strongest backbone (DINOv2 ViT-B), warm-started from the
_pb checkpoint:

  Conditioning vector (51-dim):
    - union bbox raw [4-dim]                  (raw positional signal, same as _pb)
    - rich player features [13-dim]           (from exp1: center, size, aspect, doubles)
    - rule-of-thirds distances [7-dim]        (from exp2: RoT proximity features)
    - 3x3 image stats [27-dim]               (from exp5: brightness/edges/saturation)
  → 2-layer encoder → [128-dim]
  + DINOv2 CLS [768-dim]
  → box head + angle head

Warm-start: backbone weights loaded from cropping_angle_vit_base_patch14_reg4_dinov2_pb.pt
            (already adapted to the cropping task). New conditioning encoder + heads
            trained from scratch.

Training: lr=5e-6 (backbone), lr=5e-5 (new heads), warmup=5, epochs=40, batch=8.
Loss:     same CropLoss as _pb (4-sided player clipping penalty on primary bbox).

Outputs:
    logs/exp6_combined_dinov2.log
    checkpoints/cropping_angle_vit_base_patch14_reg4_dinov2_exp6_combined.pt
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import CropLoss, box_iou_numpy

# ── constants ──────────────────────────────────────────────────────────────────
BACKBONE        = "vit_base_patch14_reg4_dinov2"
EPOCHS          = 40
BATCH_SIZE      = 8
LR_BACKBONE     = 5e-7    # very small — backbone already adapted from _pb
LR_HEAD         = 5e-5    # new conditioning encoder + heads learn faster
WARMUP_EPOCHS   = 5
GRAD_CHECKPOINT = True
ANGLE_SCALE     = 90.0
PLAYER_WEIGHT   = 0.5
ANGLE_WEIGHT    = 0.25
NUM_WORKERS     = 4

COND_DIM        = 51      # 4 (raw union) + 13 (rich) + 7 (RoT) + 27 (img stats)
COND_EMB_DIM    = 128

_UNION_BBOX_CACHE   = ROOT / "data" / "player_bboxes.json"
_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"
_PB_CKPT            = CHECKPOINTS_DIR / "cropping_angle_vit_base_patch14_reg4_dinov2_pb.pt"
_CKPT_PATH          = CHECKPOINTS_DIR / "cropping_angle_vit_base_patch14_reg4_dinov2_exp6_combined.pt"
_LOG_FILE           = ROOT / "logs" / "exp6_combined_dinov2.log"

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


# ── conditioning feature builders (copied from exp1/exp2/exp5) ─────────────────

def build_rich_features(union_bbox: torch.Tensor,
                        primary_bbox: torch.Tensor) -> torch.Tensor:
    """[B,4] union + [B,4] primary → [B, 13]  (from exp1)"""
    eps = 1e-6
    ux1, uy1, ux2, uy2 = union_bbox.unbind(1)
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
    vert_third = (cy / (1.0 / 3.0 + eps)).clamp(max=2.0) / 2.0
    area_ratio = u_area / (p_area + eps)
    has_two    = (area_ratio > 1.4).float()
    return torch.stack([
        cx, cy, uw, uh, u_area,
        uw / (uh + eps),
        dx, dy, dist,
        vert_third, u_area,
        (area_ratio - 1.0).clamp(min=0.0),
        has_two,
    ], dim=1)  # [B, 13]


def build_rot_features(player_bbox: torch.Tensor) -> torch.Tensor:
    """[B,4] primary bbox → [B, 7]  (from exp2)"""
    x1, y1, x2, y2 = player_bbox.unbind(1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    r1, r2 = 1 / 3, 2 / 3
    dx = torch.minimum(torch.abs(cx - r1), torch.abs(cx - r2))
    dy = torch.minimum(torch.abs(cy - r1), torch.abs(cy - r2))
    area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).sqrt()
    return torch.stack([
        dx, dy,
        cx - r1, cx - r2,
        cy - r1, cy - r2,
        area,
    ], dim=1)  # [B, 7]


def compute_region_stats(img: Image.Image, grid: int = 3) -> list:
    """PIL image → 27 floats (from exp5): 3×3 grid × (brightness, edge, saturation)"""
    h, w = img.height, img.width
    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = img_np.mean(axis=2)
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edges = gx + gy
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat  = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-6), 0.0)
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


# ── model ──────────────────────────────────────────────────────────────────────

class CombinedDINOv2(nn.Module):
    """
    DINOv2 ViT-B + 51-dim combined conditioning → box + angle.

    Forward: model(x, union_bbox, primary_bbox, img_stats)
      x           [B, 3, 518, 518]
      union_bbox  [B, 4]  (zeros if not detected)
      primary_bbox[B, 4]  (zeros if not detected)
      img_stats   [B, 27]
    Returns: (box [B,4], angle_norm [B])
    """
    def __init__(self, backbone: nn.Module, backbone_dim: int):
        super().__init__()
        self.backbone = backbone

        self.cond_encoder = nn.Sequential(
            nn.Linear(COND_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, COND_EMB_DIM),
            nn.ReLU(),
        )

        head_in = backbone_dim + COND_EMB_DIM  # 768 + 128 = 896

        self.box_head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                union_bbox: torch.Tensor,
                primary_bbox: torch.Tensor,
                img_stats: torch.Tensor):
        feats = self.backbone(x)                          # [B, 768]
        rich  = build_rich_features(union_bbox, primary_bbox)  # [B, 13]
        rot   = build_rot_features(primary_bbox)               # [B, 7]
        cond  = torch.cat([union_bbox, rich, rot, img_stats], dim=1)  # [B, 51]
        cond_emb = self.cond_encoder(cond)                     # [B, 128]
        combined = torch.cat([feats, cond_emb], dim=1)         # [B, 896]
        box   = self.box_head(combined)                        # [B, 4]
        angle = self.angle_head(combined).squeeze(-1)          # [B]
        return box, angle

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=enable)


def build_model() -> CombinedDINOv2:
    # Always create backbone without timm pretrained — we warm-start from _pb.
    # (timm's norm→fc_norm key rename causes a RuntimeError for DINOv2 anyway.)
    backbone = timm.create_model(BACKBONE, pretrained=False,
                                 num_classes=0, global_pool="avg")
    dim = backbone.num_features  # 768 for ViT-B
    model = CombinedDINOv2(backbone, dim)

    if _PB_CKPT.exists():
        log.info(f"  Warm-starting backbone from {_PB_CKPT.name}")
        ck = torch.load(str(_PB_CKPT), map_location="cpu", weights_only=False)
        pb_state = {k[len("backbone."):]: v
                    for k, v in ck["model_state"].items()
                    if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(pb_state, strict=False)
        log.info(f"  Backbone: {len(pb_state)-len(unexpected)} keys loaded  "
                 f"missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        log.warning(f"  _pb checkpoint not found at {_PB_CKPT}; using random init")

    return model


# ── dataset ────────────────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    def __init__(self, records: list, transform, hflip: bool = False,
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
        img = extract_thumbnail_ar(r["raw"], max_size=512)
        box = list(r["box"])
        angle_norm = r.get("angle_deg", 0.0) / ANGLE_SCALE

        ub = self.union_cache.get(r["raw"])
        union_bbox = list(ub) if ub is not None else [0.0, 0.0, 0.0, 0.0]

        pb = self.primary_cache.get(r["raw"])
        primary_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]

        # Image stats computed on PIL thumbnail before transforms
        img_stats = compute_region_stats(img, grid=3)  # 27 floats

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
            # flip image stats: swap left/right columns (col 0 ↔ col 2)
            s = img_stats
            img_stats = [
                s[6],  s[7],  s[8],   # col 2, row 0  →  col 0, row 0
                s[3],  s[4],  s[5],   # col 1 unchanged
                s[0],  s[1],  s[2],   # col 0  →  col 2
                s[15], s[16], s[17],  # col 2, row 1
                s[12], s[13], s[14],
                s[9],  s[10], s[11],
                s[24], s[25], s[26],  # col 2, row 2
                s[21], s[22], s[23],
                s[18], s[19], s[20],
            ]

        img_t     = self.transform(img)
        box_t     = torch.tensor(box,          dtype=torch.float32)
        angle_t   = torch.tensor(angle_norm,   dtype=torch.float32)
        union_t   = torch.tensor(union_bbox,   dtype=torch.float32)
        primary_t = torch.tensor(primary_bbox, dtype=torch.float32)
        stats_t   = torch.tensor(img_stats,    dtype=torch.float32)
        return img_t, box_t, angle_t, union_t, primary_t, stats_t


# ── evaluation ─────────────────────────────────────────────────────────────────

def _evaluate(model: CombinedDINOv2, loader: DataLoader,
              device: torch.device) -> dict:
    model.eval()
    all_pred, all_gt, all_angles, all_gt_angles = [], [], [], []
    with torch.no_grad():
        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes, stats in loader:
            imgs          = imgs.to(device)
            union_bboxes  = union_bboxes.to(device)
            primary_bboxes= primary_bboxes.to(device)
            stats         = stats.to(device)
            box_pred, angle_pred = model(imgs, union_bboxes, primary_bboxes, stats)
            all_pred.append(box_pred.cpu().numpy())
            all_gt.append(boxes.numpy())
            all_angles.append(angle_pred.cpu().numpy())
            all_gt_angles.append(angle_norms.numpy())
    pred_arr = np.concatenate(all_pred)
    gt_arr   = np.concatenate(all_gt)
    ious     = box_iou_numpy(pred_arr, gt_arr)
    angle_mae = float(np.abs(
        np.concatenate(all_angles) - np.concatenate(all_gt_angles)
    ).mean() * ANGLE_SCALE)
    return {
        "mean_iou":     float(ious.mean()),
        "median_iou":   float(np.median(ious)),
        "iou_gt70":     float((ious >= 0.70).mean()),
        "iou_gt80":     float((ious >= 0.80).mean()),
        "angle_mae_deg": angle_mae,
        "n":            len(ious),
    }


# ── training ───────────────────────────────────────────────────────────────────

def train() -> None:
    log.info("=" * 70)
    log.info("Experiment 6: DINOv2 ViT-B + combined conditioning (51-dim)")
    log.info(f"  backbone={BACKBONE}  epochs={EPOCHS}  batch={BATCH_SIZE}")
    log.info(f"  lr_backbone={LR_BACKBONE:.0e}  lr_head={LR_HEAD:.0e}  warmup={WARMUP_EPOCHS}")
    log.info(f"  cond_dim={COND_DIM}  cond_emb={COND_EMB_DIM}")
    log.info("=" * 70)

    with open(CROP_GT_FILE) as fh:
        all_recs = json.load(fh)
    train_recs = [r for r in all_recs if r["split"] == "train"]
    val_recs   = [r for r in all_recs if r["split"] == "val"]
    test_recs  = [r for r in all_recs if r["split"] == "test"]
    log.info(f"  train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    union_cache, primary_cache = {}, {}
    if _UNION_BBOX_CACHE.exists():
        with open(_UNION_BBOX_CACHE) as fh:
            union_cache = json.load(fh)
        log.info(f"  Union cache: {len(union_cache):,} entries")
    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            primary_cache = json.load(fh)
        log.info(f"  Primary cache: {len(primary_cache):,} entries")
    else:
        primary_cache = union_cache
        log.warning("  Primary bbox cache missing; using union as fallback")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  device={device}")

    model = build_model()
    if GRAD_CHECKPOINT:
        model.set_grad_checkpointing(True)
    model.to(device)

    data_cfg  = timm.data.resolve_model_data_config(model.backbone)
    inp_sz    = data_cfg.get("input_size", (3, 518, 518))[1]
    mean_     = tuple(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std_      = tuple(data_cfg.get("std",  (0.229, 0.224, 0.225)))
    log.info(f"  input_size={inp_sz}  norm_mean={mean_}  norm_std={std_}")

    tf_train = transforms.Compose([
        transforms.Resize((inp_sz, inp_sz)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(list(mean_), list(std_)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((inp_sz, inp_sz)),
        transforms.ToTensor(),
        transforms.Normalize(list(mean_), list(std_)),
    ])

    train_ds = CombinedDataset(train_recs, tf_train, hflip=True,
                               union_cache=union_cache, primary_cache=primary_cache)
    val_ds   = CombinedDataset(val_recs,   tf_val,   hflip=False,
                               union_cache=union_cache, primary_cache=primary_cache)
    test_ds  = CombinedDataset(test_recs,  tf_val,   hflip=False,
                               union_cache=union_cache, primary_cache=primary_cache)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Two param groups: backbone at very low lr, new heads at higher lr
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.cond_encoder.parameters()) +
        list(model.box_head.parameters()) +
        list(model.angle_head.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 0.01},
        {"params": head_params,     "lr": LR_HEAD,     "weight_decay": 0.01},
    ])

    total_epochs = EPOCHS
    if WARMUP_EPOCHS > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=WARMUP_EPOCHS)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_epochs - WARMUP_EPOCHS, 1))
        sched = SequentialLR(optimizer,
                             schedulers=[warmup_sched, cosine_sched],
                             milestones=[WARMUP_EPOCHS])
    else:
        sched = CosineAnnealingLR(optimizer, T_max=total_epochs)

    criterion = CropLoss(alpha=0.5, angle_weight=ANGLE_WEIGHT,
                         player_weight=PLAYER_WEIGHT)

    best_iou     = -1.0
    best_metrics: dict = {}
    per_epoch_val: list = []

    log.info(f"\n  Training for {total_epochs} epochs...")
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes, stats in tqdm(
                train_loader, desc=f"[exp6] ep{epoch}/{total_epochs}", leave=False):
            imgs          = imgs.to(device)
            boxes         = boxes.to(device)
            angle_norms   = angle_norms.to(device)
            union_bboxes  = union_bboxes.to(device)
            primary_bboxes= primary_bboxes.to(device)
            stats         = stats.to(device)

            optimizer.zero_grad()
            box_pred, angle_pred = model(imgs, union_bboxes, primary_bboxes, stats)
            loss = criterion(box_pred, boxes, angle_pred, angle_norms,
                             player_bbox=primary_bboxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / len(train_loader)

        val_m = _evaluate(model, val_loader, device)
        per_epoch_val.append(val_m["mean_iou"])

        log.info(
            f"  ep{epoch:02d}  loss={avg_loss:.4f}  "
            f"[val] mean_iou={val_m['mean_iou']:.4f}  "
            f"med={val_m['median_iou']:.4f}  "
            f">0.7:{val_m['iou_gt70']:.1%}  "
            f">0.8:{val_m['iou_gt80']:.1%}  "
            f"angle_mae={val_m['angle_mae_deg']:.2f}deg"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m, epoch=epoch, backbone=BACKBONE)
            torch.save({
                "epoch":          epoch,
                "backbone":       BACKBONE,
                "use_angle_head": True,
                "use_player_bbox": True,
                "model_state":    model.state_dict(),
                "metrics":        val_m,
                "input_size":     inp_sz,
                "norm_mean":      mean_,
                "norm_std":       std_,
                "angle_scale":    ANGLE_SCALE,
                "exp":            "exp6_combined",
                "cond_dim":       COND_DIM,
                "cond_emb_dim":   COND_EMB_DIM,
            }, _CKPT_PATH)
            log.info(f"      [SAVED] best val IoU={best_iou:.4f}  (ep{epoch})")

    # Test evaluation on best checkpoint
    log.info("\n" + "=" * 70)
    ck = torch.load(str(_CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = _evaluate(model, test_loader, device)

    log.info(f"TEST RESULTS  (best val ep{best_metrics['epoch']}/{EPOCHS})")
    log.info(f"  mean_iou  = {test_m['mean_iou']:.4f}")
    log.info(f"  median    = {test_m['median_iou']:.4f}")
    log.info(f"  >0.70     = {test_m['iou_gt70']:.1%}")
    log.info(f"  >0.80     = {test_m['iou_gt80']:.1%}")
    log.info(f"  angle_mae = {test_m['angle_mae_deg']:.2f}deg")
    log.info(f"\n  Best val mIoU = {best_iou:.4f}  (ep {best_metrics['epoch']}/{EPOCHS})")
    log.info(f"  vs _pb baseline: val=0.8407  test=0.8318")
    delta = test_m["mean_iou"] - 0.8318
    log.info(f"  delta vs _pb test: {delta:+.4f}")

    log.info("\nPer-epoch val IoU:")
    for ep, iou in enumerate(per_epoch_val, 1):
        mark = " <-- best" if abs(iou - best_iou) < 1e-6 else ""
        log.info(f"  ep{ep:02d}  {iou:.4f}{mark}")


if __name__ == "__main__":
    train()
