"""
Held-out test-set evaluation for full-trained exp9 crop checkpoints.

Evaluates on the 552-photo test split with the exact val-time protocol
(CropDatasetExp9, no augmentation, soft-blend head routing).

Usage:
    python eval_crop_test.py                    # evaluates exp9d_full + exp9e_full
    python eval_crop_test.py --ckpts exp9d_full
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CROP_GT_FILE
from models.cropping.model import build_exp9_model, POSE_DIM
from models.cropping.train import (
    CropDatasetExp9, _evaluate_exp8,
    _PLAYER_BBOX_CACHE, _PRIMARY_PLAYER_BBOX_CACHE, _POSE_KEYPOINT_CACHE,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["exp9d_full", "exp9e_full"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    with open(CROP_GT_FILE) as fh:
        recs = [r for r in json.load(fh) if r["split"] == args.split]
    print(f"{args.split} split: {len(recs)} records")

    caches = {}
    for name, path in (("player_bbox_cache", _PLAYER_BBOX_CACHE),
                       ("primary_bbox_cache", _PRIMARY_PLAYER_BBOX_CACHE),
                       ("pose_kpt_cache", _POSE_KEYPOINT_CACHE)):
        with open(path) as fh:
            caches[name] = json.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tag in args.ckpts:
        ckpt_path = CHECKPOINTS_DIR / f"cropping_angle_vit_large_patch14_reg4_dinov2_{tag}.pt"
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        input_size = int(ck.get("input_size", 770))

        tf_val = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        loader = DataLoader(
            CropDatasetExp9(recs, tf_val, hflip=False, geo_aug=False, **caches),
            batch_size=4, shuffle=False, num_workers=4, pin_memory=True,
        )

        model = build_exp9_model(
            pretrained=False,
            pose_dim=ck.get("pose_dim", POSE_DIM),
            cond_emb_dim=ck.get("cond_emb_dim", 128),
            dynamic_img_size=ck.get("dynamic_img_size", True),
        ).to(device)
        model.load_state_dict(ck["model_state"])

        val_m = ck.get("metrics", {})
        m = _evaluate_exp8(model, loader, device)
        print(f"\n[{tag}] ep{ck.get('epoch', '?')}  "
              f"(val_iou was {val_m.get('mean_iou', 0):.4f})")
        print(f"  TEST  mean_iou={m['mean_iou']:.4f}  median={m['median_iou']:.4f}"
              f"  >0.7:{m['iou_gt70']:.1%}  >0.8:{m['iou_gt80']:.1%}"
              f"  angle_mae={m['angle_mae_deg']:.2f}deg  n={m['n']}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
