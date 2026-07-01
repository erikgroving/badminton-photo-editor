"""
Train small / medium / large crop regression models and print a comparison table.

Small:  efficientnet_b0      ~5.3M  params  224 px  batch=32  lr=1e-4
Medium: efficientnet_b3      ~12M   params  300 px  batch=16  lr=1e-4
Large:  vit_base_patch14_reg4_dinov2  87M  518 px  batch=8   lr=1e-5  warmup=3

Pre-requisite:
    python -m data.crop_detector          # builds data/crop_gt.json

Usage:
    python sweep_crop.py                  # all three
    python sweep_crop.py --sizes small    # one size

Log: logs/sweep_crop.log
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.cropping.train import train

LOG_PATH = Path("logs/sweep_crop.jsonl")

CONFIGS = {
    "small": {
        "backbone":        "efficientnet_b0",
        "epochs":          25,
        "batch_size":      32,
        "lr":              1e-4,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
    },
    "medium": {
        "backbone":        "efficientnet_b3",
        "epochs":          25,
        "batch_size":      16,
        "lr":              1e-4,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
    },
    "large": {
        "backbone":        "vit_base_patch14_reg4_dinov2",
        "epochs":          25,
        "batch_size":      8,
        "lr":              1e-5,
        "warmup_epochs":   3,
        "grad_checkpoint": True,
    },
}


def main(sizes: list[str]) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    results: list[dict] = []
    for size in sizes:
        cfg = CONFIGS[size]
        print(f"\n{'='*65}")
        print(f"  {size.upper()}: {cfg['backbone']}")
        print("=" * 65)
        try:
            m = train(**cfg)
            m["size"] = size
            results.append(m)
            with open(LOG_PATH, "a") as fh:
                fh.write(json.dumps({"size": size, **m}) + "\n")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"size": size, "backbone": cfg["backbone"],
                            "mean_iou": 0.0, "median_iou": 0.0})

    _print_table(results)


def _print_table(results: list[dict]) -> None:
    if not results:
        return
    print(f"\n{'='*75}")
    print(f"  {'Size':<8} {'Backbone':<38} {'Mean IoU':>9} {'Median':>8} {'>0.7':>7} {'>0.8':>7}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x.get("mean_iou", 0), reverse=True):
        tm  = r.get("test_metrics", r)
        m_  = tm.get("mean_iou",   r.get("mean_iou",   0))
        med = tm.get("median_iou", r.get("median_iou", 0))
        g70 = tm.get("iou_gt70",   0)
        g80 = tm.get("iou_gt80",   0)
        mark = "  <-- best" if r is max(results, key=lambda x: x.get("mean_iou", 0)) else ""
        print(f"  {r.get('size','?'):<8} {r['backbone']:<38} {m_:>9.4f} {med:>8.4f} "
              f"{g70:>6.1%} {g80:>6.1%}{mark}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+",
        choices=["small", "medium", "large"],
        default=["small", "medium", "large"],
    )
    args = parser.parse_args()
    main(args.sizes)
