"""
Train all backbone candidates for the judge discriminator and compare.

Usage:
    python -m models.judge.sweep [--epochs N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import JUDGE_BACKBONE_CANDIDATES, JUDGE_BATCH_SIZE, JUDGE_EPOCHS, JUDGE_LR
from models.judge.train import train


def sweep(epochs: int, batch_size: int, lr: float) -> None:
    results = []
    for backbone in JUDGE_BACKBONE_CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Judge backbone: {backbone}")
        print("=" * 60)
        try:
            m = train(backbone, epochs, batch_size, lr)
            results.append(m)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"backbone": backbone, "accuracy": 0.0})

    print("\n" + "=" * 60)
    print(f"{'Backbone':<35} {'Accuracy':>10}")
    print("-" * 60)
    best = max(results, key=lambda x: x.get("accuracy", 0))
    for r in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
        marker = "  <-- best" if r is best else ""
        print(f"{r['backbone']:<35} {r.get('accuracy', 0):>10.4f}{marker}")
    print("=" * 60)
    print(f"\nBest backbone: {best['backbone']}  (accuracy={best.get('accuracy',0):.4f})")
    print("Use this backbone for GAN-style color correction training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=JUDGE_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=JUDGE_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=JUDGE_LR)
    args = parser.parse_args()
    sweep(args.epochs, args.batch_size, args.lr)
