"""
Exp 7 crop model: ViT-L DINOv2 at 770px + pose keypoints + dual portrait/landscape heads.

Usage:
    python train_crop_exp7.py
    python train_crop_exp7.py --epochs 30 --batch-size 4 --lr 5e-6
    python train_crop_exp7.py --resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.cropping.train import train_exp7

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone",        type=str,   default="vit_large_patch14_reg4_dinov2")
    ap.add_argument("--input-size",      type=int,   default=770)
    ap.add_argument("--epochs",          type=int,   default=25)
    ap.add_argument("--batch-size",      type=int,   default=4)
    ap.add_argument("--lr",              type=float, default=5e-6)
    ap.add_argument("--warmup",          type=int,   default=3)
    ap.add_argument("--ar-weight",       type=float, default=0.1,
                    help="Aspect-ratio loss weight (0 to disable)")
    ap.add_argument("--no-grad-ckpt",    action="store_true",
                    help="Disable gradient checkpointing (uses more VRAM)")
    ap.add_argument("--ckpt-tag",        type=str,   default="_exp7")
    ap.add_argument("--resume",          action="store_true")
    ap.add_argument("--gt-file",         type=str,   default=None)
    args = ap.parse_args()

    train_exp7(
        backbone        = args.backbone,
        input_size      = args.input_size,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        warmup_epochs   = args.warmup,
        grad_checkpoint = not args.no_grad_ckpt,
        ar_weight       = args.ar_weight,
        ckpt_tag        = args.ckpt_tag,
        resume          = args.resume,
        gt_file         = Path(args.gt_file) if args.gt_file else None,
    )
