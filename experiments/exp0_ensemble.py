"""
Stage 0: Zero-training prediction ensemble across all exp checkpoints.

Loads all available experiment checkpoints, runs each on the test split,
and averages (or IoU-weighted averages) the box predictions.

Reports individual model test mIoU and ensemble mIoU.

Usage:
    python experiments/exp0_ensemble.py
    python experiments/exp0_ensemble.py --weighted   # weight by val IoU
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import box_iou_numpy, CropLoss

_UNION_BBOX_CACHE   = ROOT / "data" / "player_bboxes.json"
_PRIMARY_BBOX_CACHE = ROOT / "data" / "primary_player_bboxes.json"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
ANGLE_SCALE = 90.0


# ── checkpoints to ensemble ────────────────────────────────────────────────────
# Each entry: (label, ckpt_filename)
# Set a label to None to auto-skip if ckpt not found.
CKPT_REGISTRY = [
    ("DINOv2_pb",  "cropping_angle_vit_base_patch14_reg4_dinov2_pb.pt"),
    ("exp1_rich",  "cropping_angle_efficientnet_b3_exp1.pt"),
    ("exp2_rot",   "cropping_angle_efficientnet_b3_exp2.pt"),
    ("exp3_rot_aux", "cropping_efficientnet_b3_exp3_rot.pt"),
    ("exp4_spatial", "cropping_angle_efficientnet_b3_exp4.pt"),
    ("exp5_stats", "cropping_efficientnet_b3_exp5_imgstats.pt"),
    ("exp6_combined", "cropping_angle_vit_base_patch14_reg4_dinov2_exp6_combined.pt"),
]


# ── generic model loader ───────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    """
    Load ANY crop model checkpoint by reconstructing the architecture from
    the saved metadata. Handles baseline CropRegressor, exp1-5 custom classes,
    and exp6 CombinedDINOv2.
    """
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    backbone_name = ck.get("backbone", "efficientnet_b3")
    use_angle     = ck.get("use_angle_head", False)
    use_pb        = ck.get("use_player_bbox", False)
    exp_tag       = ck.get("exp", "")
    inp_sz        = ck.get("input_size", 224)
    norm_mean     = ck.get("norm_mean", _IMAGENET_MEAN)
    norm_std      = ck.get("norm_std",  _IMAGENET_STD)
    val_iou       = ck.get("metrics", {}).get("mean_iou", 0.0)
    saved_epoch   = ck.get("epoch", "?")

    # Reconstruct the model. For exp6 we need the custom class; for all others
    # the standard CropRegressor from model.py works.
    if exp_tag == "exp6_combined":
        from experiments.exp6_combined_dinov2 import CombinedDINOv2
        backbone = timm.create_model(backbone_name, pretrained=False,
                                     num_classes=0, global_pool="avg")
        model = CombinedDINOv2(backbone, backbone.num_features)
    elif ck.get("exp", "").startswith("exp1") or "rich" in str(ckpt_path):
        from experiments.exp1_rich_features import EnrichedCropModel
        model = EnrichedCropModel(backbone=backbone_name, pretrained=False)
    elif ck.get("exp", "").startswith("exp2") or "exp2" in str(ckpt_path):
        from experiments.exp2_rule_of_thirds import build_rot_model
        model = build_rot_model(backbone=backbone_name, pretrained=False)
    elif "exp3" in str(ckpt_path):
        # exp3 uses the same architecture as exp2 (RoT conditioning)
        try:
            from experiments.exp3_rule_of_thirds import build_rot_model as build3
            model = build3(backbone=backbone_name, pretrained=False)
        except Exception:
            from experiments.exp2_rule_of_thirds import build_rot_model
            model = build_rot_model(backbone=backbone_name, pretrained=False)
    elif "exp4" in str(ckpt_path):
        try:
            from experiments.exp4_spatial_features import build_spatial_model
            model = build_spatial_model(backbone=backbone_name, pretrained=False)
        except Exception:
            from models.cropping.model import build_crop_model
            model = build_crop_model(backbone=backbone_name, pretrained=False,
                                     use_angle_head=use_angle, use_player_bbox=use_pb)
    elif "exp5" in str(ckpt_path):
        from experiments.exp5_image_stats import build_stats_model
        model = build_stats_model(backbone=backbone_name, pretrained=False)
    else:
        from models.cropping.model import build_crop_model
        model = build_crop_model(backbone=backbone_name, pretrained=False,
                                 use_angle_head=use_angle, use_player_bbox=use_pb)

    model.load_state_dict(ck["model_state"])
    model.eval()
    model.to(device)

    return model, {
        "backbone":  backbone_name,
        "exp":       exp_tag,
        "inp_sz":    inp_sz,
        "norm_mean": norm_mean,
        "norm_std":  norm_std,
        "use_pb":    use_pb,
        "use_angle": use_angle,
        "val_iou":   val_iou,
        "epoch":     saved_epoch,
        "ckpt_path": ckpt_path,
    }


# ── generic inference ──────────────────────────────────────────────────────────

def predict_boxes(model, meta: dict, test_recs: list,
                  union_cache: dict, primary_cache: dict,
                  device: torch.device) -> np.ndarray:
    """Run model on all test records; return [N, 4] box predictions."""
    inp_sz = meta["inp_sz"]
    tf = transforms.Compose([
        transforms.Resize((inp_sz, inp_sz)),
        transforms.ToTensor(),
        transforms.Normalize(list(meta["norm_mean"]), list(meta["norm_std"])),
    ])

    exp_tag = meta["exp"]
    needs_stats = (exp_tag == "exp6_combined") or ("exp5" in str(meta["ckpt_path"]))
    needs_two   = "exp1" in str(meta["ckpt_path"])
    needs_rot   = ("exp2" in str(meta["ckpt_path"]) or "exp3" in str(meta["ckpt_path"]))

    all_preds = []

    for r in test_recs:
        img  = extract_thumbnail_ar(r["raw"], max_size=512)
        img_t = tf(img).unsqueeze(0).to(device)

        ub = union_cache.get(r["raw"])
        pb = primary_cache.get(r["raw"])
        union_t   = torch.tensor([ub if ub else [0]*4], dtype=torch.float32).to(device)
        primary_t = torch.tensor([pb if pb else [0]*4], dtype=torch.float32).to(device)

        with torch.no_grad():
            if exp_tag == "exp6_combined":
                from experiments.exp6_combined_dinov2 import compute_region_stats
                stats = torch.tensor([compute_region_stats(img)], dtype=torch.float32).to(device)
                box, _ = model(img_t, union_t, primary_t, stats)
            elif needs_stats:
                from experiments.exp5_image_stats import compute_region_stats as crs
                stats = torch.tensor([crs(img)], dtype=torch.float32).to(device)
                box = model(img_t, primary_t, stats)
            elif needs_two:
                from experiments.exp1_rich_features import build_rich_features
                rich = build_rich_features(union_t, primary_t)
                box = model(img_t, rich)
            elif needs_rot:
                from experiments.exp2_rule_of_thirds import compute_rot_features
                rot_feats = torch.cat([union_t, compute_rot_features(primary_t)], dim=1)
                box = model(img_t, rot_feats)
            elif meta["use_pb"]:
                out = model(img_t, union_t)
                box = out[0] if isinstance(out, tuple) else out
            else:
                out = model(img_t)
                box = out[0] if isinstance(out, tuple) else out

        all_preds.append(box.squeeze(0).cpu().numpy())

    return np.stack(all_preds)  # [N, 4]


# ── main ───────────────────────────────────────────────────────────────────────

def main(weighted: bool = False) -> None:
    print("=" * 70)
    print("  Exp0: Prediction Ensemble")
    print(f"  mode={'IoU-weighted' if weighted else 'uniform average'}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    with open(CROP_GT_FILE) as fh:
        all_recs = json.load(fh)
    test_recs = [r for r in all_recs if r["split"] == "test"]
    gt_boxes  = np.array([r["box"] for r in test_recs])
    print(f"  test samples: {len(test_recs)}")

    union_cache, primary_cache = {}, {}
    if _UNION_BBOX_CACHE.exists():
        with open(_UNION_BBOX_CACHE) as fh:
            union_cache = json.load(fh)
    if _PRIMARY_BBOX_CACHE.exists():
        with open(_PRIMARY_BBOX_CACHE) as fh:
            primary_cache = json.load(fh)

    model_preds = []   # list of (label, val_iou, preds [N,4])
    individual_results = []

    for label, ckpt_name in CKPT_REGISTRY:
        ckpt_path = CHECKPOINTS_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"  [{label}] SKIP — checkpoint not found")
            continue

        print(f"\n  [{label}] Loading {ckpt_name} ...", flush=True)
        try:
            model, meta = load_model_from_ckpt(ckpt_path, device)
        except Exception as e:
            print(f"    FAILED to load: {e}")
            continue

        print(f"    backbone={meta['backbone']}  val_iou={meta['val_iou']:.4f}  ep={meta['epoch']}")
        preds = predict_boxes(model, meta, test_recs, union_cache, primary_cache, device)
        ious  = box_iou_numpy(preds, gt_boxes)
        test_miou = float(ious.mean())
        print(f"    test  mean_iou={test_miou:.4f}  median={float(np.median(ious)):.4f}  "
              f">0.7:{float((ious>=0.7).mean()):.1%}  >0.8:{float((ious>=0.8).mean()):.1%}")

        model_preds.append((label, meta["val_iou"], preds))
        individual_results.append((label, test_miou, float(np.median(ious)),
                                   float((ious>=0.7).mean()), float((ious>=0.8).mean())))

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    if len(model_preds) < 2:
        print("\n  Not enough models for ensemble.")
        return

    # ── ensemble ───────────────────────────────────────────────────────────────
    print(f"\n  Building ensemble from {len(model_preds)} models...")
    all_p = np.stack([p for _, _, p in model_preds])  # [M, N, 4]

    # Uniform average
    uniform_preds = all_p.mean(axis=0)
    uniform_ious  = box_iou_numpy(uniform_preds, gt_boxes)
    print(f"\n  [Uniform avg] mean_iou={uniform_ious.mean():.4f}  "
          f"median={np.median(uniform_ious):.4f}  "
          f">0.7:{(uniform_ious>=0.7).mean():.1%}  "
          f">0.8:{(uniform_ious>=0.8).mean():.1%}")

    # Weighted average by val IoU
    weights = np.array([v for _, v, _ in model_preds])
    weights = weights / weights.sum()
    weighted_preds = (all_p * weights[:, None, None]).sum(axis=0)
    weighted_ious  = box_iou_numpy(weighted_preds, gt_boxes)
    print(f"  [Weighted avg] mean_iou={weighted_ious.mean():.4f}  "
          f"median={np.median(weighted_ious):.4f}  "
          f">0.7:{(weighted_ious>=0.7).mean():.1%}  "
          f">0.8:{(weighted_ious>=0.8).mean():.1%}")

    print("\n" + "=" * 70)
    print(f"  {'Model':<20} {'Test mIoU':>10} {'Median':>8} {'>0.7':>6} {'>0.8':>6}")
    print("-" * 70)
    for label, miou, med, g70, g80 in sorted(individual_results, key=lambda x: -x[1]):
        print(f"  {label:<20} {miou:>10.4f} {med:>8.4f} {g70:>5.1%} {g80:>5.1%}")
    print("-" * 70)
    print(f"  {'Ensemble (uniform)':<20} {uniform_ious.mean():>10.4f} "
          f"{np.median(uniform_ious):>8.4f} {(uniform_ious>=0.7).mean():>5.1%} "
          f"{(uniform_ious>=0.8).mean():>5.1%}")
    print(f"  {'Ensemble (weighted)':<20} {weighted_ious.mean():>10.4f} "
          f"{np.median(weighted_ious):>8.4f} {(weighted_ious>=0.7).mean():>5.1%} "
          f"{(weighted_ious>=0.8).mean():>5.1%}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()
    main(weighted=args.weighted)
