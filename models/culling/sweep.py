"""
Train all backbone candidates for the culling model and print a comparison table.

Usage:
    python -m models.culling.sweep [--epochs N] [--fn-weight F]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHECKPOINTS_DIR, CULL_BACKBONE_CANDIDATES, CULL_BATCH_SIZE, CULL_EPOCHS, CULL_FBETA, CULL_FN_WEIGHT, CULL_LR
from models.culling.train import train


# Some backbones need a smaller batch size to fit in VRAM at 512x512
_BATCH_SIZE_OVERRIDES: dict[str, int] = {
    "convnext_small": 4,
}


def sweep(epochs: int, batch_size: int, fn_weight: float, lr: float) -> None:
    results = []
    for backbone in CULL_BACKBONE_CANDIDATES:
        safe = backbone.replace("/", "_")
        ckpt = CHECKPOINTS_DIR / f"culling_{safe}.pt"

        if ckpt.exists():
            print(f"\n{'='*70}")
            print(f"Backbone: {backbone}  [SKIP — checkpoint exists]")
            print("=" * 70)
            try:
                import torch
                ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
                m  = dict(ck.get("metrics", {}), backbone=backbone)
                results.append(m)
            except Exception:
                results.append({"backbone": backbone, "asym_cost": float("inf"),
                                "recall": 0.0, "selection_rate": 0.0, f"f{CULL_FBETA:.0f}": 0.0})
            continue

        effective_batch = _BATCH_SIZE_OVERRIDES.get(backbone, batch_size)
        print(f"\n{'='*70}")
        print(f"Backbone: {backbone}  (batch_size={effective_batch})")
        print("=" * 70)
        try:
            m = train(backbone, epochs, effective_batch, fn_weight, lr)
            results.append(m)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback; traceback.print_exc()
            results.append({"backbone": backbone, "asym_cost": float("inf"), "recall": 0.0,
                            "selection_rate": 0.0, f"f{CULL_FBETA:.0f}": 0.0})

    beta_key = f"f{CULL_FBETA:.0f}"
    print("\n" + "=" * 90)
    print(f"{'Backbone':<35} {'Asym Cost':>10} {'Recall':>8} {'Selected%':>10} {beta_key:>8}")
    print("-" * 90)
    best = min(results, key=lambda x: x.get("asym_cost", float("inf")))
    for r in sorted(results, key=lambda x: x.get("asym_cost", float("inf"))):
        marker = "  <-- best" if r is best else ""
        sel_pct = r.get("selection_rate", 0) * 100
        print(
            f"{r['backbone']:<35} {r.get('asym_cost', 0):>10.1f}"
            f" {r.get('recall', 0):>8.3f} {sel_pct:>9.1f}%"
            f" {r.get(beta_key, 0):>8.4f}{marker}"
        )
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=CULL_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=CULL_BATCH_SIZE)
    parser.add_argument("--fn-weight",  type=float, default=CULL_FN_WEIGHT)
    parser.add_argument("--lr",         type=float, default=CULL_LR)
    args = parser.parse_args()
    sweep(args.epochs, args.batch_size, args.fn_weight, args.lr)
