"""
Experiment 3: Player Pose Keypoints as Conditioning
====================================================
Hypothesis: Knowing WHERE the player's body parts are (feet, hands, head)
gives the model precise crop boundary information. Feet keypoints (COCO kp 15 & 16,
indices 45-50 in the flat list) directly encode where the bottom of the crop should be.

Architecture:
  - efficientnet_b3 backbone (pretrained, num_classes=0)
  - Pose keypoint encoder: Linear(51 -> 64) + ReLU
  - Primary bbox encoder (for player-coverage loss): same as baseline
  - Combined head: Linear(backbone_feats + 64 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256, 4) -> Sigmoid

Settings:
  - 10 epochs, batch_size=16, lr=1e-4
  - CosineAnnealingLR (no warmup)
  - Player-coverage loss (hinge penalty on primary bbox clipping)

Output:
  - Log:        logs/exp3_pose.log
  - Checkpoint: checkpoints/cropping_angle_efficientnet_b3_exp3.pt

Run:
    python experiments/exp3_pose_keypoints.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import CropLoss, box_iou_numpy

# ── Paths ──────────────────────────────────────────────────────────────────────
_POSE_CACHE          = ROOT / "data" / "pose_keypoints.json"
_PRIMARY_BBOX_CACHE  = ROOT / "data" / "primary_player_bboxes.json"
_LOG_FILE            = ROOT / "logs" / "exp3_pose.log"
_CKPT_FILE           = CHECKPOINTS_DIR / "cropping_angle_efficientnet_b3_exp3.pt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BACKBONE        = "efficientnet_b3"
EPOCHS          = 10
BATCH_SIZE      = 16
LR              = 1e-4
POSE_EMB_DIM    = 64
POSE_INPUT_DIM  = 51   # 17 keypoints × 3 (x, y, conf)
HEAD_HIDDEN     = 256
DROPOUT         = 0.3

_IMAGENET_MEAN  = (0.485, 0.456, 0.406)
_IMAGENET_STD   = (0.229, 0.224, 0.225)
_EXTRACT_SIZE   = 512


# ── Logging ───────────────────────────────────────────────────────────────────
class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ── Model ─────────────────────────────────────────────────────────────────────
def build_pose_model(backbone: str = BACKBONE, pretrained: bool = True) -> nn.Module:
    """
    Crop regressor with pose keypoint conditioning.

    Forward signature:
        model(img, pose_kps, primary_bbox=None)
          img:          [B, 3, H, W]
          pose_kps:     [B, 51] flat normalized keypoints
          primary_bbox: [B, 4] for player-coverage loss (not used in forward)

    Returns:
        box: [B, 4] normalized crop prediction via Sigmoid
    """
    backbone_model = timm.create_model(
        backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    in_features = backbone_model.num_features
    head_in = in_features + POSE_EMB_DIM

    class PoseCropRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_model

            # Pose encoder: 51 -> 64
            # The key insight: feet keypoints (indices 45-50 = kp15 & kp16)
            # directly encode where the bottom of the crop should be.
            self.pose_encoder = nn.Sequential(
                nn.Linear(POSE_INPUT_DIM, POSE_EMB_DIM),
                nn.ReLU(),
            )

            # Box head: (backbone_feats + pose_emb) -> 4 crop coords
            self.box_head = nn.Sequential(
                nn.Linear(head_in, HEAD_HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HEAD_HIDDEN, 4),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor, pose_kps: torch.Tensor) -> torch.Tensor:
            feats    = self.backbone(x)
            pose_emb = self.pose_encoder(pose_kps)
            feats    = torch.cat([feats, pose_emb], dim=1)
            return self.box_head(feats)

        def set_grad_checkpointing(self, enable: bool = True) -> None:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=enable)

    return PoseCropRegressor()


# ── Dataset ───────────────────────────────────────────────────────────────────
class PoseCropDataset(Dataset):
    """
    Extends the baseline CropDataset with pose keypoints.

    Returns: (img_t, box_t, pose_kps_t, primary_bbox_t)
    """
    def __init__(
        self,
        records: list[dict],
        transform,
        hflip: bool = False,
        pose_cache: dict | None = None,
        primary_bbox_cache: dict | None = None,
    ):
        self.records            = records
        self.transform          = transform
        self.hflip              = hflip
        self.pose_cache         = pose_cache or {}
        self.primary_bbox_cache = primary_bbox_cache or {}

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _zero_kps() -> list[float]:
        return [0.0] * POSE_INPUT_DIM

    def __getitem__(self, idx: int):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=_EXTRACT_SIZE)
        box = list(r["box"])   # [x1, y1, x2, y2] in [0, 1]

        # Pose keypoints
        kps = self.pose_cache.get(r["raw"])
        pose_kps = list(kps) if kps is not None else self._zero_kps()

        # Primary bbox for clipping penalty
        pb = self.primary_bbox_cache.get(r["raw"])
        primary_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]

        if self.hflip and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Flip box
            x1, y1, x2, y2 = box
            box = [1.0 - x2, y1, 1.0 - x1, y2]

            # Flip primary bbox
            if pb is not None:
                px1, py1, px2, py2 = primary_bbox
                primary_bbox = [1.0 - px2, py1, 1.0 - px1, py2]

            # Flip keypoints: x -> 1 - x, swap left/right pairs
            # COCO left/right pairs: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)
            _LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            kps_arr = list(pose_kps)  # 51 floats
            # Flip x coords (every 3rd starting at 0)
            for ki in range(17):
                base = ki * 3
                kps_arr[base] = 1.0 - kps_arr[base]
            # Swap left/right joints
            for (la, ra) in _LR_PAIRS:
                ba, br = la * 3, ra * 3
                kps_arr[ba:ba+3], kps_arr[br:br+3] = kps_arr[br:br+3], kps_arr[ba:ba+3]
            pose_kps = kps_arr

        img_t         = self.transform(img)
        box_t         = torch.tensor(box,         dtype=torch.float32)
        pose_kps_t    = torch.tensor(pose_kps,    dtype=torch.float32)
        primary_bbox_t = torch.tensor(primary_bbox, dtype=torch.float32)
        return img_t, box_t, pose_kps_t, primary_bbox_t


# ── Evaluation ────────────────────────────────────────────────────────────────
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for imgs, boxes, pose_kps, _primary in loader:
            box_pred = model(imgs.to(device), pose_kps.to(device))
            all_pred.append(box_pred.cpu().numpy())
            all_gt.append(boxes.numpy())
    pred_arr = np.concatenate(all_pred)
    gt_arr   = np.concatenate(all_gt)
    ious     = box_iou_numpy(pred_arr, gt_arr)
    return {
        "mean_iou":   float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou_gt50":   float((ious >= 0.50).mean()),
        "iou_gt70":   float((ious >= 0.70).mean()),
        "iou_gt80":   float((ious >= 0.80).mean()),
        "n":          len(ious),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    # ── Redirect stdout to tee ──
    tee = Tee(_LOG_FILE)
    sys.stdout = tee

    print(f"=" * 70)
    print(f"Exp 3: Pose Keypoints Conditioning — {BACKBONE}")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    print(f"  log    -> {_LOG_FILE}")
    print(f"  ckpt   -> {_CKPT_FILE}")
    print(f"=" * 70)
    t_start = time.time()

    # ── Load GT ──
    if not CROP_GT_FILE.exists():
        raise FileNotFoundError(f"GT file not found: {CROP_GT_FILE}")
    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    print(f"  train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    # ── Load caches ──
    pose_cache: dict = {}
    if _POSE_CACHE.exists():
        with open(_POSE_CACHE) as fh:
            pose_cache = json.load(fh)
        n_covered = sum(1 for r in all_records if pose_cache.get(r["raw"]) is not None)
        pose_detected = sum(
            1 for r in all_records
            if pose_cache.get(r["raw"]) is not None and
               any(v > 0.8 for v in (pose_cache[r["raw"]] or [])[2::3])
        )
        print(f"  Pose cache: {len(pose_cache):,} entries  "
              f"({n_covered}/{len(all_records)} GT covered, "
              f"~{pose_detected} with high-conf YOLO detections)")
    else:
        print(f"  WARNING: pose cache not found at {_POSE_CACHE}")
        print(f"  Run:  python data/cache_pose_keypoints.py")
        print(f"  Continuing with zero keypoints (model will learn from bbox info only)")

    primary_cache: dict = {}
    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            primary_cache = json.load(fh)
        print(f"  Primary bbox cache: {len(primary_cache):,} entries (used for coverage loss)")

    # ── Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model = build_pose_model(backbone=BACKBONE, pretrained=True).to(device)

    # ── Transforms ──
    data_cfg   = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean  = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    norm_std   = tuple(data_cfg.get("std",  _IMAGENET_STD))
    print(f"  input_size={input_size}  norm_mean={norm_mean}  norm_std={norm_std}")

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
        PoseCropDataset(train_recs, tf_train, hflip=True,
                        pose_cache=pose_cache,
                        primary_bbox_cache=primary_cache),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        PoseCropDataset(val_recs, tf_val,
                        pose_cache=pose_cache,
                        primary_bbox_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        PoseCropDataset(test_recs, tf_val,
                        pose_cache=pose_cache,
                        primary_bbox_cache=primary_cache),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )

    criterion = CropLoss(alpha=0.5, angle_weight=0.0, player_weight=0.5, player_margin=0.0)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    best_iou     = -1.0
    best_metrics: dict = {}
    epoch_log: list[dict] = []

    print()
    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val IoU':>8}  {'Med IoU':>8}  "
          f"{'IoU>0.7':>7}  {'IoU>0.8':>7}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, boxes, pose_kps, primary_bboxes in tqdm(
                train_loader, desc=f"ep{epoch}/{EPOCHS}", leave=False):
            imgs           = imgs.to(device)
            boxes          = boxes.to(device)
            pose_kps       = pose_kps.to(device)
            primary_bboxes = primary_bboxes.to(device)

            optimizer.zero_grad()
            box_pred = model(imgs, pose_kps)
            # Pass primary_bboxes for player-coverage hinge penalty
            loss = criterion(box_pred, boxes, player_bbox=primary_bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_m = _evaluate(model, val_loader, device)
        row   = dict(val_m, epoch=epoch, loss=avg_loss)
        epoch_log.append(row)

        print(
            f"{epoch:>5}  {avg_loss:>8.4f}  {val_m['mean_iou']:>8.4f}  "
            f"{val_m['median_iou']:>8.4f}  {val_m['iou_gt70']:>7.1%}  "
            f"{val_m['iou_gt80']:>7.1%}"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m, backbone=BACKBONE, epoch=epoch)
            torch.save({
                "epoch":      epoch,
                "backbone":   BACKBONE,
                "model_type": "pose_keypoints",
                "model_state": model.state_dict(),
                "metrics":    val_m,
                "input_size": input_size,
                "norm_mean":  norm_mean,
                "norm_std":   norm_std,
                "pose_emb_dim": POSE_EMB_DIM,
            }, _CKPT_FILE)
            print(f"  --> Best val IoU={best_iou:.4f} saved to {_CKPT_FILE.name}")

    # ── Test evaluation on best checkpoint ──
    ck = torch.load(str(_CKPT_FILE), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m = _evaluate(model, test_loader, device)
    best_metrics["test_metrics"] = test_m

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"TEST RESULTS (best epoch={best_metrics['epoch']})")
    print(f"  mean_iou   = {test_m['mean_iou']:.4f}")
    print(f"  median_iou = {test_m['median_iou']:.4f}")
    print(f"  iou_gt70   = {test_m['iou_gt70']:.1%}")
    print(f"  iou_gt80   = {test_m['iou_gt80']:.1%}")
    print(f"  n_test     = {test_m['n']}")
    print(f"  elapsed    = {elapsed/60:.1f} min")
    print("=" * 60)
    print()
    print("Val IoU per epoch (for comparison):")
    for row in epoch_log:
        flag = " <-- best" if row["epoch"] == best_metrics["epoch"] else ""
        print(f"  ep{row['epoch']:02d}  loss={row['loss']:.4f}  "
              f"val_iou={row['mean_iou']:.4f}  "
              f"iou>0.8={row['iou_gt80']:.1%}{flag}")

    # ── Save epoch log as JSON alongside checkpoint ──
    log_json = _CKPT_FILE.with_suffix(".log.json")
    with open(log_json, "w") as fh:
        json.dump({"epoch_log": epoch_log, "best_metrics": best_metrics}, fh, indent=2)
    print(f"\nEpoch log saved -> {log_json}")

    sys.stdout = tee._stdout
    tee.close()
    return best_metrics


if __name__ == "__main__":
    train()
