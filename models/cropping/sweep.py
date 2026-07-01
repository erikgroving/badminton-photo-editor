"""
Train all backbone candidates for the crop model and print a comparison table.

Usage:
    python -m models.cropping.sweep [--epochs N] [--batch-size N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CROP_BACKBONE_CANDIDATES, CROP_BATCH_SIZE, CROP_EPOCHS, CROP_LR
from models.cropping.train import train


def sweep(epochs: int, batch_size: int, lr: float) -> None:
    results = []
    for backbone in CROP_BACKBONE_CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Backbone: {backbone}")
        print("=" * 60)
        try:
            metrics = train(backbone, epochs, batch_size, lr)
            results.append(metrics)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"backbone": backbone, "median_iou": 0.0, "mean_iou": 0.0})

    print("\n" + "=" * 60)
    print(f"{'Backbone':<35} {'Median IoU':>12} {'Mean IoU':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.get("median_iou", 0), reverse=True):
        marker = "  <-- best" if r == max(results, key=lambda x: x.get("median_iou", 0)) else ""
        print(f"{r['backbone']:<35} {r.get('median_iou', 0):>12.4f} {r.get('mean_iou', 0):>10.4f}{marker}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CROP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CROP_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CROP_LR)
    args = parser.parse_args()
    sweep(args.epochs, args.batch_size, args.lr)
