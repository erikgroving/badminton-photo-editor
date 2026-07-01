"""
Sweep false-negative penalty weights to characterise the recall / selection-rate
trade-off for the culling model.

Usage:
    python -m models.culling.fn_weight_sweep
    python -m models.culling.fn_weight_sweep --backbone efficientnet_b3
    python -m models.culling.fn_weight_sweep --fn-weights 2 5 10 15 20 30

Prints a comparison table:

  FN weight | Recall  | Selection | Precision |   F2   | Asym cost
  ----------+---------+-----------+-----------+--------+-----------
        2   |  71.3%  |   14.2%   |   48.7%   | 0.612  |    984
  ...

A low fn_weight keeps selection rate tight (higher precision) at the cost of
missing more of Jay's shots.  A high fn_weight guarantees almost everything
Jay kept is included, but the photographer still has to manually cull
more false positives afterward.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHECKPOINTS_DIR, CULL_BATCH_SIZE, CULL_EPOCHS, CULL_FN_WEIGHT, CULL_LR
from models.culling.train import _ckpt_path, train

DEFAULT_FN_WEIGHTS = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   type=str,   default="efficientnet_b0",
                        help="Backbone to use for all fn-weight runs (fastest = efficientnet_b0)")
    parser.add_argument("--fn-weights", type=float, nargs="+", default=DEFAULT_FN_WEIGHTS,
                        metavar="W", help="List of FN penalty weights to sweep")
    parser.add_argument("--epochs",     type=int,   default=CULL_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=CULL_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=CULL_LR)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip weights whose checkpoint already exists (default: on)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    results = []
    for w in args.fn_weights:
        ckpt = _ckpt_path(args.backbone, w)
        if args.skip_existing and ckpt.exists():
            import torch
            print(f"\n[fn_weight={w:.0f}] Checkpoint exists — loading: {ckpt.name}")
            ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            m = ck.get("metrics", {})
            results.append({"fn_weight": w, **m})
            continue

        print(f"\n{'='*65}")
        print(f"  fn_weight = {w:.0f}x  ({args.backbone})")
        print(f"{'='*65}")
        m = train(args.backbone, args.epochs, args.batch_size, w, args.lr)
        results.append({"fn_weight": w, **m})

    _print_table(results)


def _print_table(results: list[dict]) -> None:
    if not results:
        return
    print("\n" + "="*75)
    print("  FN-Weight Sweep — Recall vs Selection Rate Trade-off")
    print("="*75)
    hdr = f"{'FN wt':>7}  {'Recall':>8}  {'Selectn':>8}  {'Precis':>8}  {'F2':>8}  {'AsymCost':>10}"
    print(hdr)
    print("-" * 75)
    for r in sorted(results, key=lambda x: x["fn_weight"]):
        recall    = r.get("recall", 0)
        sel       = r.get("selection_rate", 0)
        prec      = recall / sel if sel > 0 else 0.0
        f2        = r.get("f2", 0)
        cost      = r.get("asym_cost", 0)
        w         = r["fn_weight"]
        print(f"{w:>7.0f}  {recall:>7.1%}  {sel:>7.1%}  {prec:>7.1%}  {f2:>8.4f}  {cost:>10.0f}")
    print("="*75)
    print()
    print("Recall     = % of Jay's kept photos that the model also keeps")
    print("Selection  = % of ALL photos the model marks as keep (lower = less work for you)")
    print("Precision  = % of model-selected photos that were Jay's actual picks")
    print("AsymCost   = FN x fn_weight + FP  (lower is better given your weight choice)")


if __name__ == "__main__":
    main()
