"""
Train the crop regression model (box-only or box+angle).

Input:  raw thumbnails (aspect-ratio-preserving), squished to model input size.
Target: [x1, y1, x2, y2] normalized to [0, 1] + optional angle_deg/90 regression.

Usage:
    python -m models.cropping.train --backbone efficientnet_b0
    python -m models.cropping.train --backbone efficientnet_b0 --angle-head

Checkpoint: checkpoints/cropping[_angle]_<backbone>[<ckpt_tag>].pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, CROP_BATCH_SIZE, CROP_EPOCHS, CROP_GT_FILE, CROP_LR,
)
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import (
    ANGLE_SCALE, CropLoss, box_iou_numpy, build_crop_model,
    CombinedDINOv2Exp7, build_exp7_model, flip_pose_kpts, POSE_DIM,
    CombinedDINOv2Exp8, build_exp8_model,
    CombinedDINOv2Exp9, build_exp9_model,
    compute_region_stats,
)

_PLAYER_BBOX_CACHE         = Path(__file__).parent.parent.parent / "data" / "player_bboxes.json"
_PRIMARY_PLAYER_BBOX_CACHE = Path(__file__).parent.parent.parent / "data" / "primary_player_bboxes.json"
_POSE_KEYPOINT_CACHE       = Path(__file__).parent.parent.parent / "data" / "pose_keypoints.json"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_EXTRACT_SIZE  = 512


def _ckpt_path(backbone: str, use_angle_head: bool, ckpt_tag: str = "") -> Path:
    head_suffix = "_angle" if use_angle_head else ""
    return CHECKPOINTS_DIR / f"cropping{head_suffix}_{backbone.replace('/', '_')}{ckpt_tag}.pt"


class CropDataset(Dataset):
    def __init__(self, records: list[dict], transform, hflip: bool = False,
                 player_bbox_cache: dict | None = None,
                 primary_bbox_cache: dict | None = None):
        self.records            = records
        self.transform          = transform
        self.hflip              = hflip
        self.player_bbox_cache  = player_bbox_cache  or {}  # union bbox for model conditioning
        self.primary_bbox_cache = primary_bbox_cache or {}  # primary player for loss penalty

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r   = self.records[idx]
        img = extract_thumbnail_ar(r["raw"], max_size=_EXTRACT_SIZE)
        box = list(r["box"])                                   # [x1, y1, x2, y2] in [0, 1]
        angle_norm = r.get("angle_deg", 0.0) / ANGLE_SCALE    # normalize: /90

        # Union bbox for model conditioning (all detected persons)
        ub = self.player_bbox_cache.get(r["raw"])
        union_bbox = list(ub) if ub is not None else [0.0, 0.0, 0.0, 0.0]

        # Primary player bbox for clipping penalty (largest detected person)
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

        img_t         = self.transform(img)
        box_t         = torch.tensor(box,          dtype=torch.float32)
        angle_t       = torch.tensor(angle_norm,   dtype=torch.float32)
        union_bbox_t  = torch.tensor(union_bbox,   dtype=torch.float32)
        primary_bbox_t = torch.tensor(primary_bbox, dtype=torch.float32)
        return img_t, box_t, angle_t, union_bbox_t, primary_bbox_t


def _evaluate(model: torch.nn.Module, loader: DataLoader,
              device: torch.device, use_angle_head: bool) -> dict:
    model.eval()
    all_pred, all_gt, all_angle_pred, all_angle_gt = [], [], [], []

    with torch.no_grad():
        for imgs, boxes, angle_norms, union_bboxes, _primary in loader:
            out = model(imgs.to(device), union_bboxes.to(device))
            if use_angle_head:
                box_pred, angle_pred = out
                all_angle_pred.append(angle_pred.cpu().numpy())
                all_angle_gt.append(angle_norms.numpy())
            else:
                box_pred = out
            all_pred.append(box_pred.cpu().numpy())
            all_gt.append(boxes.numpy())

    pred_arr = np.concatenate(all_pred)
    gt_arr   = np.concatenate(all_gt)
    ious     = box_iou_numpy(pred_arr, gt_arr)

    result = {
        "mean_iou":   float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou_gt50":   float((ious >= 0.50).mean()),
        "iou_gt70":   float((ious >= 0.70).mean()),
        "iou_gt80":   float((ious >= 0.80).mean()),
        "n":          len(ious),
    }

    if use_angle_head and all_angle_pred:
        pred_deg = np.concatenate(all_angle_pred) * ANGLE_SCALE
        true_deg = np.concatenate(all_angle_gt)   * ANGLE_SCALE
        result["angle_mae_deg"] = float(np.mean(np.abs(pred_deg - true_deg)))

    return result


def train(backbone: str, epochs: int, batch_size: int, lr: float,
          warmup_epochs: int = 0, grad_checkpoint: bool = False,
          use_angle_head: bool = False, use_player_bbox: bool = False,
          ckpt_tag: str = "", resume: bool = False,
          gt_file: Path | None = None) -> dict:
    gt_path = Path(gt_file) if gt_file else CROP_GT_FILE
    if not gt_path.exists():
        raise FileNotFoundError(
            f"GT file not found: {gt_path}\n"
            f"  For SIFT GT:  python -m data.crop_detector\n"
            f"  For YOLO GT:  python -m data.build_yolo_crop_gt"
        )

    with open(gt_path) as fh:
        all_records = json.load(fh)

    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    mode_parts = []
    if use_angle_head:   mode_parts.append("angle")
    if use_player_bbox:  mode_parts.append("player-bbox")
    mode_str = "+".join(mode_parts) if mode_parts else "box-only"
    print(f"[{backbone}] {mode_str}  train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    player_bbox_cache: dict = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        n_covered = sum(1 for r in all_records if player_bbox_cache.get(r["raw"]) is not None)
        print(f"  Union player bbox cache: {len(player_bbox_cache):,} entries  "
              f"({n_covered}/{len(all_records)} GT raws covered)")
    else:
        print(f"  Warning: no player bbox cache at {_PLAYER_BBOX_CACHE}. "
              f"Run: python data/cache_player_bboxes.py")

    primary_bbox_cache: dict = {}
    if _PRIMARY_PLAYER_BBOX_CACHE.exists():
        with open(_PRIMARY_PLAYER_BBOX_CACHE) as fh:
            primary_bbox_cache = json.load(fh)
        n_primary = sum(1 for r in all_records if primary_bbox_cache.get(r["raw"]) is not None)
        print(f"  Primary player bbox cache: {len(primary_bbox_cache):,} entries  "
              f"({n_primary}/{len(all_records)} GT raws covered)")
    else:
        # Fallback: use union bbox for clipping penalty (singles = same as primary; doubles = approximate)
        primary_bbox_cache = player_bbox_cache
        print(f"  Note: primary bbox cache not found — using union bbox for clipping penalty."
              f" Run: python data/cache_player_bboxes.py")

    angles = [r.get("angle_deg", 0.0) for r in all_records]
    n_portrait = sum(1 for a in angles if a >= 45.0)
    print(f"  Portrait (>=45 deg): {n_portrait}/{len(all_records)} = {n_portrait/len(all_records):.1%}  "
          f"angle range: [{min(angles):.1f}, {max(angles):.1f}] deg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_crop_model(backbone=backbone, pretrained=True,
                              use_angle_head=use_angle_head,
                              use_player_bbox=use_player_bbox)
    if grad_checkpoint:
        model.set_grad_checkpointing(enable=True)
    model = model.to(device)

    data_cfg   = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean  = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
    norm_std   = tuple(data_cfg.get("std",  _IMAGENET_STD))

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

    nw = min(4, batch_size)
    train_loader = DataLoader(
        CropDataset(train_recs, tf_train, hflip=True,
                    player_bbox_cache=player_bbox_cache,
                    primary_bbox_cache=primary_bbox_cache),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        CropDataset(val_recs, tf_val,
                    player_bbox_cache=player_bbox_cache,
                    primary_bbox_cache=primary_bbox_cache),
        batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        CropDataset(test_recs, tf_val,
                    player_bbox_cache=player_bbox_cache,
                    primary_bbox_cache=primary_bbox_cache),
        batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True,
    )

    criterion = CropLoss(alpha=0.5, angle_weight=0.25,
                         player_weight=0.5, player_margin=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if warmup_epochs > 0 and warmup_epochs < epochs:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        sched  = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sched  = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = _ckpt_path(backbone, use_angle_head, ckpt_tag)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch  = 1
    best_iou     = -1.0
    best_metrics: dict = {}

    if resume and ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        model.to(device)
        start_epoch  = ck["epoch"] + 1
        best_iou     = ck["metrics"]["mean_iou"]
        best_metrics = dict(ck["metrics"], backbone=backbone, epoch=ck["epoch"])
        remaining    = epochs - (start_epoch - 1)
        print(f"  Resumed from ep{ck['epoch']} (val IoU={best_iou:.4f}); "
              f"continuing for {remaining} more epochs")
        if remaining > 0:
            sched = CosineAnnealingLR(optimizer, T_max=max(remaining, 1))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, boxes, angle_norms, union_bboxes, primary_bboxes in tqdm(
                train_loader, desc=f"[{backbone}] ep{epoch}/{epochs}", leave=False):
            imgs          = imgs.to(device)
            boxes         = boxes.to(device)
            angle_norms   = angle_norms.to(device)
            union_bboxes  = union_bboxes.to(device)
            primary_bboxes = primary_bboxes.to(device)
            optimizer.zero_grad()
            out = model(imgs, union_bboxes)           # conditioning: union (all players)
            if use_angle_head:
                box_pred, angle_pred = out
                loss = criterion(box_pred, boxes, angle_pred, angle_norms,
                                 player_bbox=primary_bboxes)  # penalty: primary player
            else:
                loss = criterion(out, boxes, player_bbox=primary_bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        sched.step()

        val_m   = _evaluate(model, val_loader, device, use_angle_head)
        mae_str = (f"  angle_mae={val_m['angle_mae_deg']:.2f}deg"
                   if "angle_mae_deg" in val_m else "")
        print(
            f"  ep{epoch:02d}"
            f"  loss={total_loss/len(train_loader):.4f}"
            f"  [val] mean_iou={val_m['mean_iou']:.4f}"
            f"  med={val_m['median_iou']:.4f}"
            f"  >0.7:{val_m['iou_gt70']:.1%}"
            f"  >0.8:{val_m['iou_gt80']:.1%}"
            f"{mae_str}"
        )

        if val_m["mean_iou"] > best_iou:
            best_iou     = val_m["mean_iou"]
            best_metrics = dict(val_m, backbone=backbone, epoch=epoch)
            torch.save({
                "epoch":           epoch,
                "backbone":        backbone,
                "use_angle_head":  use_angle_head,
                "use_player_bbox": use_player_bbox,
                "model_state":     model.state_dict(),
                "metrics":         val_m,
                "input_size":      input_size,
                "norm_mean":       norm_mean,
                "norm_std":        norm_std,
                "angle_scale":     ANGLE_SCALE,
            }, ckpt_path)
            print(f"    [OK] Saved best (mean_iou={best_iou:.4f}{mae_str})")

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    test_m  = _evaluate(model, test_loader, device, use_angle_head)
    mae_str = (f"  angle_mae={test_m['angle_mae_deg']:.2f}deg"
               if "angle_mae_deg" in test_m else "")
    print(
        f"[{backbone}] TEST"
        f"  mean_iou={test_m['mean_iou']:.4f}"
        f"  median={test_m['median_iou']:.4f}"
        f"  >0.7:{test_m['iou_gt70']:.1%}"
        f"  >0.8:{test_m['iou_gt80']:.1%}"
        f"{mae_str}"
    )
    best_metrics["test_metrics"] = test_m
    return best_metrics


# ── Exp 7: pose keypoints + dual heads + AR loss + ViT-L 770px ───────────────

class CropDatasetExp7(Dataset):
    """Dataset for exp7: adds pose keypoints + img_stats conditioning."""
    def __init__(self, records: list[dict], transform, hflip: bool = False,
                 player_bbox_cache: dict | None = None,
                 primary_bbox_cache: dict | None = None,
                 pose_kpt_cache: dict | None = None):
        self.records            = records
        self.transform          = transform
        self.hflip              = hflip
        self.player_bbox_cache  = player_bbox_cache  or {}
        self.primary_bbox_cache = primary_bbox_cache or {}
        self.pose_kpt_cache     = pose_kpt_cache     or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r          = self.records[idx]
        img        = extract_thumbnail_ar(r["raw"], max_size=_EXTRACT_SIZE)
        box        = list(r["box"])
        angle_norm = r.get("angle_deg", 0.0) / ANGLE_SCALE

        ub           = self.player_bbox_cache.get(r["raw"])
        union_bbox   = list(ub) if ub is not None else [0.0, 0.0, 0.0, 0.0]
        pb           = self.primary_bbox_cache.get(r["raw"])
        primary_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]

        # Pose: [x0,y0,c0, x1,y1,c1, ...] → take x,y only (every 3rd value dropped)
        raw_kpts = self.pose_kpt_cache.get(r["raw"])
        if raw_kpts is not None:
            kpts = [raw_kpts[i] for i in range(len(raw_kpts)) if i % 3 != 2]
        else:
            kpts = [0.0] * POSE_DIM

        # Compute region stats before any augmentation (on native AR thumbnail)
        img_stats = compute_region_stats(img)

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
            kpts = flip_pose_kpts(kpts)

        return (
            self.transform(img),
            torch.tensor(box,          dtype=torch.float32),
            torch.tensor(angle_norm,   dtype=torch.float32),
            torch.tensor(union_bbox,   dtype=torch.float32),
            torch.tensor(primary_bbox, dtype=torch.float32),
            torch.tensor(kpts,         dtype=torch.float32),
            torch.tensor(img_stats,    dtype=torch.float32),
        )


def _evaluate_exp7(model: CombinedDINOv2Exp7, loader: DataLoader,
                   device: torch.device) -> dict:
    model.eval()
    all_pred, all_gt, all_angle_pred, all_angle_gt = [], [], [], []
    with torch.no_grad():
        for imgs, boxes, angle_norms, union_t, primary_t, pose_t, stats_t in loader:
            box_pred, angle_pred = model(
                imgs.to(device), union_t.to(device), primary_t.to(device),
                stats_t.to(device), pose_t.to(device),
            )
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
        "iou_gt50":      float((ious >= 0.50).mean()),
        "iou_gt70":      float((ious >= 0.70).mean()),
        "iou_gt80":      float((ious >= 0.80).mean()),
        "angle_mae_deg": float(np.mean(np.abs(pred_deg - true_deg))),
        "n":             len(ious),
    }


def train_exp7(backbone: str = "vit_large_patch14_reg4_dinov2",
               input_size: int = 770,
               epochs: int = 25,
               batch_size: int = 4,
               lr: float = 5e-6,
               warmup_epochs: int = 3,
               grad_checkpoint: bool = True,
               ar_weight: float = 0.1,
               ckpt_tag: str = "_exp7",
               resume: bool = False,
               gt_file: Path | None = None) -> dict:

    gt_path = Path(gt_file) if gt_file else CROP_GT_FILE
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    with open(gt_path) as fh:
        all_records = json.load(fh)

    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    print(f"[exp7 {backbone}] train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")
    print(f"  input={input_size}  lr={lr}  epochs={epochs}  ar_weight={ar_weight}")

    player_bbox_cache: dict = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        print(f"  player_bboxes: {len(player_bbox_cache):,} entries")

    primary_bbox_cache: dict = {}
    if _PRIMARY_PLAYER_BBOX_CACHE.exists():
        with open(_PRIMARY_PLAYER_BBOX_CACHE) as fh:
            primary_bbox_cache = json.load(fh)

    pose_kpt_cache: dict = {}
    if _POSE_KEYPOINT_CACHE.exists():
        with open(_POSE_KEYPOINT_CACHE) as fh:
            pose_kpt_cache = json.load(fh)
        print(f"  pose_keypoints: {len(pose_kpt_cache):,} entries  (zeros for missing)")
    else:
        print(f"  WARNING: {_POSE_KEYPOINT_CACHE} not found — pose conditioning will be zeros")

    tf_train = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ds_kw = dict(player_bbox_cache=player_bbox_cache,
                 primary_bbox_cache=primary_bbox_cache,
                 pose_kpt_cache=pose_kpt_cache)
    train_loader = DataLoader(
        CropDatasetExp7(train_recs, tf_train, hflip=True, **ds_kw),
        batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        CropDatasetExp7(val_recs,   tf_val,   hflip=False, **ds_kw),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        CropDatasetExp7(test_recs,  tf_val,   hflip=False, **ds_kw),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  workers=4")

    model = build_exp7_model(backbone=backbone, pretrained=True,
                              dynamic_img_size=(input_size != 518)).to(device)
    if grad_checkpoint:
        model.set_grad_checkpointing(True)

    ckpt_path = CHECKPOINTS_DIR / f"cropping_angle_{backbone.replace('/', '_')}{ckpt_tag}.pt"

    best_val_iou = 0.0
    start_epoch  = 0
    best_metrics: dict = {}

    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        best_val_iou = ck.get("metrics", {}).get("mean_iou", 0.0)
        start_epoch  = ck.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch-1}  best_val_iou={best_val_iou:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(train_loader)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        sched_cos    = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.01)
        scheduler    = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cos],
                                    milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    criterion = CropLoss(alpha=0.5, angle_weight=0.25, player_weight=0.5,
                         ar_weight=ar_weight)

    n_portrait = sum(1 for r in train_recs if r.get("angle_deg", 0.0) >= 45.0)
    print(f"  portrait in train: {n_portrait}/{len(train_recs)} ({n_portrait/len(train_recs):.1%})")

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        for imgs, boxes, angle_norms, union_t, primary_t, pose_t, stats_t in tqdm(
                train_loader, desc=f"Ep {epoch+1}/{start_epoch+epochs}", leave=False):
            imgs       = imgs.to(device)
            boxes      = boxes.to(device)
            angle_norms = angle_norms.to(device)
            union_t    = union_t.to(device)
            primary_t  = primary_t.to(device)
            pose_t     = pose_t.to(device)
            stats_t    = stats_t.to(device)

            optimizer.zero_grad()
            box_pred, angle_pred = model(imgs, union_t, primary_t, stats_t, pose_t)
            loss = criterion(box_pred, boxes, angle_pred, angle_norms, primary_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m    = _evaluate_exp7(model, val_loader, device)
        print(f"  ep={epoch+1}  loss={avg_loss:.4f}"
              f"  val_iou={val_m['mean_iou']:.4f}  median={val_m['median_iou']:.4f}"
              f"  >0.8:{val_m['iou_gt80']:.1%}"
              f"  angle_mae={val_m['angle_mae_deg']:.2f}deg")

        if val_m["mean_iou"] > best_val_iou:
            best_val_iou = val_m["mean_iou"]
            best_metrics = val_m
            torch.save({
                "exp":             "exp7",
                "backbone":        backbone,
                "input_size":      input_size,
                "dynamic_img_size": (input_size != 518),
                "cond_dim":        model.cond_dim,
                "cond_emb_dim":    model.cond_emb_dim,
                "pose_dim":        POSE_DIM,
                "ar_weight":       ar_weight,
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "metrics":         val_m,
            }, ckpt_path)
            print(f"    >> saved  mean_iou={best_val_iou:.4f}")

    # Final test evaluation
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state"])
    test_m = _evaluate_exp7(model, test_loader, device)
    print(f"\n[exp7 {backbone}] TEST"
          f"  mean_iou={test_m['mean_iou']:.4f}"
          f"  median={test_m['median_iou']:.4f}"
          f"  >0.7:{test_m['iou_gt70']:.1%}"
          f"  >0.8:{test_m['iou_gt80']:.1%}"
          f"  angle_mae={test_m['angle_mae_deg']:.2f}deg")
    best_metrics["test_metrics"] = test_m
    return best_metrics


# ── Exp 8: spatial patch features + hard head routing ─────────────────────────

def _evaluate_exp8(model: CombinedDINOv2Exp8, loader: DataLoader,
                   device: torch.device) -> dict:
    """Evaluate exp8 model — same as exp7 but is_portrait=None (soft blend at inference)."""
    model.eval()
    all_pred, all_gt, all_angle_pred, all_angle_gt = [], [], [], []
    with torch.no_grad():
        for imgs, boxes, angle_norms, union_t, primary_t, pose_t, stats_t in loader:
            box_pred, angle_pred = model(
                imgs.to(device), union_t.to(device), primary_t.to(device),
                stats_t.to(device), pose_t.to(device),
                is_portrait=None,   # soft blend at inference
            )
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
        "iou_gt50":      float((ious >= 0.50).mean()),
        "iou_gt70":      float((ious >= 0.70).mean()),
        "iou_gt80":      float((ious >= 0.80).mean()),
        "angle_mae_deg": float(np.mean(np.abs(pred_deg - true_deg))),
        "n":             len(ious),
    }


def train_exp8(backbone: str = "vit_large_patch14_reg4_dinov2",
               input_size: int = 770,
               epochs: int = 25,
               batch_size: int = 4,
               lr: float = 5e-6,
               warmup_epochs: int = 3,
               grad_checkpoint: bool = True,
               ar_weight: float = 0.1,
               ckpt_tag: str = "_exp8",
               resume: bool = False,
               gt_file: Path | None = None) -> dict:

    gt_path = Path(gt_file) if gt_file else CROP_GT_FILE
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    with open(gt_path) as fh:
        all_records = json.load(fh)

    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    test_recs  = [r for r in all_records if r["split"] == "test"]
    print(f"[exp8 {backbone}] train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")
    print(f"  input={input_size}  lr={lr}  epochs={epochs}  ar_weight={ar_weight}")
    print(f"  features: CLS+patch_mean spatial  hard head routing during training")

    player_bbox_cache: dict = {}
    if _PLAYER_BBOX_CACHE.exists():
        with open(_PLAYER_BBOX_CACHE) as fh:
            player_bbox_cache = json.load(fh)
        print(f"  player_bboxes: {len(player_bbox_cache):,} entries")

    primary_bbox_cache: dict = {}
    if _PRIMARY_PLAYER_BBOX_CACHE.exists():
        with open(_PRIMARY_PLAYER_BBOX_CACHE) as fh:
            primary_bbox_cache = json.load(fh)

    pose_kpt_cache: dict = {}
    if _POSE_KEYPOINT_CACHE.exists():
        with open(_POSE_KEYPOINT_CACHE) as fh:
            pose_kpt_cache = json.load(fh)
        print(f"  pose_keypoints: {len(pose_kpt_cache):,} entries")

    tf_train = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ds_kw = dict(player_bbox_cache=player_bbox_cache,
                 primary_bbox_cache=primary_bbox_cache,
                 pose_kpt_cache=pose_kpt_cache)
    train_loader = DataLoader(
        CropDatasetExp7(train_recs, tf_train, hflip=True, **ds_kw),
        batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        CropDatasetExp7(val_recs,   tf_val,   hflip=False, **ds_kw),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        CropDatasetExp7(test_recs,  tf_val,   hflip=False, **ds_kw),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  workers=4")

    model = build_exp8_model(backbone=backbone, pretrained=True,
                              dynamic_img_size=(input_size != 518)).to(device)
    if grad_checkpoint:
        model.set_grad_checkpointing(True)

    ckpt_path = CHECKPOINTS_DIR / f"cropping_angle_{backbone.replace('/', '_')}{ckpt_tag}.pt"

    best_val_iou = 0.0
    start_epoch  = 0

    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        best_val_iou = ck.get("metrics", {}).get("mean_iou", 0.0)
        start_epoch  = ck.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch-1}  best_val_iou={best_val_iou:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(train_loader)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        sched_cos    = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.01)
        scheduler    = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cos],
                                    milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    criterion = CropLoss(alpha=0.5, angle_weight=0.25, player_weight=0.0,
                         ar_weight=ar_weight)

    best_metrics: dict = {}
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        for imgs, boxes, angle_norms, union_t, primary_t, pose_t, stats_t in tqdm(
                train_loader, desc=f"Ep {epoch+1}/{start_epoch+epochs}", leave=False):
            imgs        = imgs.to(device)
            boxes       = boxes.to(device)
            angle_norms_d = angle_norms.to(device)
            union_t     = union_t.to(device)
            primary_t   = primary_t.to(device)
            pose_t      = pose_t.to(device)
            stats_t     = stats_t.to(device)

            # Hard routing: portrait = angle_norm >= 0.5 (i.e. angle >= 45 deg)
            is_portrait = (angle_norms_d >= 0.5)

            optimizer.zero_grad()
            box_pred, angle_pred = model(imgs, union_t, primary_t, stats_t, pose_t,
                                         is_portrait=is_portrait)
            loss = criterion(box_pred, boxes, angle_pred, angle_norms_d, primary_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m    = _evaluate_exp8(model, val_loader, device)
        print(f"  ep={epoch+1}  loss={avg_loss:.4f}"
              f"  val_iou={val_m['mean_iou']:.4f}  median={val_m['median_iou']:.4f}"
              f"  >0.8:{val_m['iou_gt80']:.1%}"
              f"  angle_mae={val_m['angle_mae_deg']:.2f}deg")

        if val_m["mean_iou"] > best_val_iou:
            best_val_iou = val_m["mean_iou"]
            best_metrics = val_m
            torch.save({
                "exp":             "exp8",
                "backbone":        backbone,
                "input_size":      input_size,
                "dynamic_img_size": (input_size != 518),
                "cond_dim":        model.cond_dim,
                "cond_emb_dim":    model.cond_emb_dim,
                "pose_dim":        POSE_DIM,
                "ar_weight":       ar_weight,
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "metrics":         val_m,
            }, ckpt_path)
            print(f"    >> saved  mean_iou={best_val_iou:.4f}")

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state"])
    test_m = _evaluate_exp8(model, test_loader, device)
    print(f"\n[exp8 {backbone}] TEST"
          f"  mean_iou={test_m['mean_iou']:.4f}"
          f"  median={test_m['median_iou']:.4f}"
          f"  >0.7:{test_m['iou_gt70']:.1%}"
          f"  >0.8:{test_m['iou_gt80']:.1%}"
          f"  angle_mae={test_m['angle_mae_deg']:.2f}deg")
    best_metrics["test_metrics"] = test_m
    return best_metrics


# ── Exp 9: pilot variants (a=arch, b=opt, c=aug) on a bug-fixed dataset ────────

POSE_CONF_THRESH = 0.3   # keypoints below this feed garbage x,y — zero them


class CropDatasetExp9(Dataset):
    """Exp7 dataset with three bug fixes + optional geometric augmentation:

    1. img_stats computed AFTER all augmentation (exp7/8 computed them before
       hflip, so half the training samples had mirrored-image/unmirrored-stats).
    2. hflip skipped for tilted samples (|angle| >2° off both 0° and 90°) —
       mirroring negates tilt sign, which exp7/8 didn't account for.
    3. Pose keypoints with confidence < POSE_CONF_THRESH zeroed (cache stores
       [x,y,conf]×17; exp7/8 dropped conf but kept the garbage x,y).

    geo_aug: random scale/translate crop (region guaranteed to contain the GT
    box + margin) with box/bbox/keypoint remap.  Simulates framing variation.
    """
    def __init__(self, records: list[dict], transform, hflip: bool = False,
                 geo_aug: bool = False,
                 player_bbox_cache: dict | None = None,
                 primary_bbox_cache: dict | None = None,
                 pose_kpt_cache: dict | None = None):
        self.records            = records
        self.transform          = transform
        self.hflip              = hflip
        self.geo_aug            = geo_aug
        self.player_bbox_cache  = player_bbox_cache  or {}
        self.primary_bbox_cache = primary_bbox_cache or {}
        self.pose_kpt_cache     = pose_kpt_cache     or {}

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_kpts(raw_kpts: list | None) -> list[float]:
        if raw_kpts is None:
            return [0.0] * POSE_DIM
        kpts = []
        for i in range(N_KPTS := POSE_DIM // 2):
            x, y, c = raw_kpts[i*3], raw_kpts[i*3+1], raw_kpts[i*3+2]
            if c < POSE_CONF_THRESH:
                kpts.extend([0.0, 0.0])
            else:
                kpts.extend([x, y])
        return kpts

    @staticmethod
    def _geo_aug(img: Image.Image, box: list, union: list, primary: list,
                 kpts: list) -> tuple:
        """Random scale/translate crop containing the GT box; remaps all coords.
        Returns inputs unchanged when no valid region exists."""
        import random as _rnd
        m  = 0.02
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        s  = _rnd.uniform(max(bw + 2*m, bh + 2*m, 0.80), 1.0)
        if s >= 0.999:
            return img, box, union, primary, kpts
        x_lo, x_hi = max(0.0, box[2] + m - s), min(1.0 - s, box[0] - m)
        y_lo, y_hi = max(0.0, box[3] + m - s), min(1.0 - s, box[1] - m)
        if x_hi < x_lo or y_hi < y_lo:      # GT box too close to an edge
            return img, box, union, primary, kpts
        x0 = _rnd.uniform(x_lo, x_hi)
        y0 = _rnd.uniform(y_lo, y_hi)

        W, H = img.size
        img  = img.crop((round(x0*W), round(y0*H),
                         round((x0+s)*W), round((y0+s)*H)))

        def _remap_box(b: list, clamp: bool) -> list:
            r = [(b[0]-x0)/s, (b[1]-y0)/s, (b[2]-x0)/s, (b[3]-y0)/s]
            if clamp:
                r = [min(max(v, 0.0), 1.0) for v in r]
                if r[2] - r[0] <= 0.0 or r[3] - r[1] <= 0.0:
                    return [0.0, 0.0, 0.0, 0.0]
            return r

        box     = _remap_box(box, clamp=False)          # guaranteed inside
        union   = _remap_box(union, clamp=True)   if union   != [0.0]*4 else union
        primary = _remap_box(primary, clamp=True) if primary != [0.0]*4 else primary

        new_kpts = []
        for i in range(POSE_DIM // 2):
            x, y = kpts[i*2], kpts[i*2+1]
            if x == 0.0 and y == 0.0:
                new_kpts.extend([0.0, 0.0])
                continue
            rx, ry = (x - x0) / s, (y - y0) / s
            if 0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0:
                new_kpts.extend([rx, ry])
            else:                                       # left the visible region
                new_kpts.extend([0.0, 0.0])
        return img, box, union, primary, new_kpts

    def __getitem__(self, idx: int):
        r          = self.records[idx]
        img        = extract_thumbnail_ar(r["raw"], max_size=_EXTRACT_SIZE)
        box        = list(r["box"])
        angle_deg  = r.get("angle_deg", 0.0)
        angle_norm = angle_deg / ANGLE_SCALE

        ub           = self.player_bbox_cache.get(r["raw"])
        union_bbox   = list(ub) if ub is not None else [0.0, 0.0, 0.0, 0.0]
        pb           = self.primary_bbox_cache.get(r["raw"])
        primary_bbox = list(pb) if pb is not None else [0.0, 0.0, 0.0, 0.0]
        kpts         = self._load_kpts(self.pose_kpt_cache.get(r["raw"]))

        if self.geo_aug and torch.rand(1).item() < 0.5:
            img, box, union_bbox, primary_bbox, kpts = self._geo_aug(
                img, box, union_bbox, primary_bbox, kpts)

        # hflip negates tilt sign — skip flipping the ~6% of tilted samples
        is_tilted = not (abs(angle_deg) <= 2.0 or abs(angle_deg - 90.0) <= 2.0)
        if self.hflip and not is_tilted and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x1, y1, x2, y2 = box
            box = [1.0 - x2, y1, 1.0 - x1, y2]
            if union_bbox != [0.0]*4:
                ux1, uy1, ux2, uy2 = union_bbox
                union_bbox = [1.0 - ux2, uy1, 1.0 - ux1, uy2]
            if primary_bbox != [0.0]*4:
                px1, py1, px2, py2 = primary_bbox
                primary_bbox = [1.0 - px2, py1, 1.0 - px1, py2]
            kpts = flip_pose_kpts(kpts)
            # flip_pose_kpts mirrors zeroed (invisible) kpts to (1.0, 0.0) — restore
            kpts = [0.0 if (kpts[i//2*2] == 1.0 and kpts[i//2*2+1] == 0.0) else kpts[i]
                    for i in range(len(kpts))]

        # stats on the FINAL augmented image (exp7/8 computed them pre-flip)
        img_stats = compute_region_stats(img)

        return (
            self.transform(img),
            torch.tensor(box,          dtype=torch.float32),
            torch.tensor(angle_norm,   dtype=torch.float32),
            torch.tensor(union_bbox,   dtype=torch.float32),
            torch.tensor(primary_bbox, dtype=torch.float32),
            torch.tensor(kpts,         dtype=torch.float32),
            torch.tensor(img_stats,    dtype=torch.float32),
        )


def train_exp9(variant: str = "a",
               backbone: str = "vit_large_patch14_reg4_dinov2",
               input_size: int = 770,
               epochs: int = 8,
               batch_size: int = 4,
               lr: float = 5e-6,
               head_lr_mult: float = 10.0,
               warmup_epochs: int = 1,
               grad_checkpoint: bool = True,
               ckpt_suffix: str = "",
               gt_file: Path | None = None) -> dict:
    """Pilot variants on the bug-fixed CropDatasetExp9 (all share fixes 1–3):

    a: exp9 architecture — multi-layer feature fusion + attentive pooling
    b: exp8 architecture + differential LR (heads ×head_lr_mult) + CIoU loss
    c: exp8 architecture + geometric scale/translate augmentation
    d: a + b + c combined (full-training candidate)
    e: a + b (no geo aug) — isolates the augmentation against d

    No test-set evaluation here — pilots are ranked on val only; the test set
    stays untouched until the winner is fully trained.
    """
    assert variant in ("a", "b", "c", "d", "e"), f"unknown variant {variant!r}"

    gt_path = Path(gt_file) if gt_file else CROP_GT_FILE
    with open(gt_path) as fh:
        all_records = json.load(fh)
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]

    desc = {
        "a": "arch: multi-layer fusion + attentive pooling",
        "b": f"opt: head LR x{head_lr_mult:g} + CIoU",
        "c": "aug: geometric scale/translate",
        "d": f"combined: fusion+attnpool + head LR x{head_lr_mult:g} + CIoU + geo aug",
        "e": f"combined: fusion+attnpool + head LR x{head_lr_mult:g} + CIoU (no geo aug)",
    }[variant]
    print(f"[exp9{variant} {backbone}] {desc}")
    print(f"  train={len(train_recs):,}  val={len(val_recs):,}  "
          f"input={input_size}  lr={lr}  epochs={epochs}")
    print(f"  dataset fixes: stats-after-aug, tilt-safe hflip, "
          f"pose conf<{POSE_CONF_THRESH} zeroed")

    caches = {}
    for name, path in (("player_bbox_cache", _PLAYER_BBOX_CACHE),
                       ("primary_bbox_cache", _PRIMARY_PLAYER_BBOX_CACHE),
                       ("pose_kpt_cache", _POSE_KEYPOINT_CACHE)):
        caches[name] = {}
        if path.exists():
            with open(path) as fh:
                caches[name] = json.load(fh)
    print(f"  caches: bbox={len(caches['player_bbox_cache']):,}  "
          f"primary={len(caches['primary_bbox_cache']):,}  "
          f"pose={len(caches['pose_kpt_cache']):,}")

    tf_train = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_loader = DataLoader(
        CropDatasetExp9(train_recs, tf_train, hflip=True,
                        geo_aug=(variant in ("c", "d")), **caches),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        CropDatasetExp9(val_recs, tf_val, hflip=False, geo_aug=False, **caches),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    build = build_exp9_model if variant in ("a", "d", "e") else build_exp8_model
    model = build(backbone=backbone, pretrained=True,
                  dynamic_img_size=(input_size != 518)).to(device)
    if grad_checkpoint:
        model.set_grad_checkpointing(True)

    if variant in ("b", "d", "e"):
        backbone_params = list(model.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": lr},
             {"params": head_params,     "lr": lr * head_lr_mult}],
            weight_decay=1e-4)
        criterion = CropLoss(alpha=0.5, angle_weight=0.25, player_weight=0.0,
                             ar_weight=0.0, use_ciou=True)   # CIoU subsumes AR term
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = CropLoss(alpha=0.5, angle_weight=0.25, player_weight=0.0,
                             ar_weight=0.1)

    total_steps = epochs * len(train_loader)
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(train_loader)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.01),
            ],
            milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    ckpt_path = CHECKPOINTS_DIR / f"cropping_angle_{backbone.replace('/', '_')}_exp9{variant}{ckpt_suffix}.pt"
    best_val_iou = 0.0
    best_metrics: dict = {}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, boxes, angle_norms, union_t, primary_t, pose_t, stats_t in tqdm(
                train_loader, desc=f"[9{variant}] Ep {epoch+1}/{epochs}", leave=False):
            imgs        = imgs.to(device)
            boxes       = boxes.to(device)
            angle_norms_d = angle_norms.to(device)
            union_t     = union_t.to(device)
            primary_t   = primary_t.to(device)
            pose_t      = pose_t.to(device)
            stats_t     = stats_t.to(device)
            is_portrait = (angle_norms_d >= 0.5)

            optimizer.zero_grad()
            box_pred, angle_pred = model(imgs, union_t, primary_t, stats_t, pose_t,
                                         is_portrait=is_portrait)
            loss = criterion(box_pred, boxes, angle_pred, angle_norms_d, primary_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m    = _evaluate_exp8(model, val_loader, device)
        print(f"  ep={epoch+1}  loss={avg_loss:.4f}"
              f"  val_iou={val_m['mean_iou']:.4f}  median={val_m['median_iou']:.4f}"
              f"  >0.8:{val_m['iou_gt80']:.1%}"
              f"  angle_mae={val_m['angle_mae_deg']:.2f}deg", flush=True)

        if val_m["mean_iou"] > best_val_iou:
            best_val_iou = val_m["mean_iou"]
            best_metrics = val_m
            torch.save({
                "exp":             f"exp9{variant}",
                "backbone":        backbone,
                "input_size":      input_size,
                "dynamic_img_size": (input_size != 518),
                "cond_dim":        model.cond_dim,
                "cond_emb_dim":    model.cond_emb_dim,
                "pose_dim":        POSE_DIM,
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "metrics":         val_m,
            }, ckpt_path)
            print(f"    >> saved  mean_iou={best_val_iou:.4f}", flush=True)

    print(f"\n[exp9{variant}] BEST val_iou={best_val_iou:.4f}  "
          f"median={best_metrics.get('median_iou', 0):.4f}  "
          f"angle_mae={best_metrics.get('angle_mae_deg', 0):.2f}deg")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",        type=str,   default="efficientnet_b0")
    parser.add_argument("--epochs",          type=int,   default=CROP_EPOCHS)
    parser.add_argument("--batch-size",      type=int,   default=CROP_BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=CROP_LR)
    parser.add_argument("--warmup",          type=int,   default=0)
    parser.add_argument("--angle-head",      action="store_true")
    parser.add_argument("--player-bbox",     action="store_true",
                        help="Condition model on YOLO player union bbox (requires player_bboxes.json)")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--ckpt-tag",        type=str,   default="")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--gt-file",         type=str,   default=None,
                        help="Path to GT JSON (default: crop_gt.json)")
    args = parser.parse_args()
    train(args.backbone, args.epochs, args.batch_size, args.lr,
          warmup_epochs=args.warmup, grad_checkpoint=args.grad_checkpoint,
          use_angle_head=args.angle_head, use_player_bbox=args.player_bbox,
          ckpt_tag=args.ckpt_tag, resume=args.resume, gt_file=args.gt_file)
