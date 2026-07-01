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
)

_PLAYER_BBOX_CACHE         = Path(__file__).parent.parent.parent / "data" / "player_bboxes.json"
_PRIMARY_PLAYER_BBOX_CACHE = Path(__file__).parent.parent.parent / "data" / "primary_player_bboxes.json"

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
