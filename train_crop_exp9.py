"""
Exp 9 pilots — three 8-epoch variants on a bug-fixed dataset, to pick one
for a full training run:

    a: multi-layer feature fusion (blocks 5/11/17/23) + attentive pooling
    b: exp8 architecture + differential head LR (x10) + CIoU loss
    c: exp8 architecture + geometric scale/translate augmentation

All variants share the dataset fixes: img_stats computed after augmentation,
hflip skipped for tilted samples, pose keypoints with conf<0.3 zeroed.

Usage:
    python train_crop_exp9.py --variant a
    python train_crop_exp9.py --variant b --epochs 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.cropping.train import train_exp9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",      type=str,   required=True, choices=["a", "b", "c", "d", "e"])
    ap.add_argument("--backbone",     type=str,   default="vit_large_patch14_reg4_dinov2")
    ap.add_argument("--input-size",   type=int,   default=770)
    ap.add_argument("--epochs",       type=int,   default=8)
    ap.add_argument("--batch-size",   type=int,   default=4)
    ap.add_argument("--lr",           type=float, default=5e-6)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--warmup",       type=int,   default=1)
    ap.add_argument("--ckpt-suffix",  type=str,   default="")
    ap.add_argument("--gt-file",      type=str,   default=None)
    args = ap.parse_args()

    train_exp9(
        variant      = args.variant,
        backbone     = args.backbone,
        input_size   = args.input_size,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        head_lr_mult = args.head_lr_mult,
        warmup_epochs= args.warmup,
        ckpt_suffix  = args.ckpt_suffix,
        gt_file      = Path(args.gt_file) if args.gt_file else None,
    )
