"""
Train crop models WITH continuous angle regression head.

Each model predicts:
  - [x1, y1, x2, y2] box (sigmoid, [0,1], in raw thumbnail space) — WHERE to crop
  - angle_deg / 90 (linear, regression) — HOW MUCH to rotate before cropping

Loss = 0.75 × (0.5×SmoothL1 + 0.5×IoU)  +  0.25 × SmoothL1(angle/90)

GT shows real ±5–13° tilt corrections in 2.5% of images (high SIFT confidence),
so binary portrait/landscape is insufficient — we regress the actual angle.

Checkpoints: checkpoints/cropping_angle_<backbone>[<tag>].pt
Log: logs/sweep_crop_angle.jsonl

Pre-requisite: data/crop_gt.json must exist (run: python -m data.crop_detector)

Usage:
    python sweep_crop_with_rotation.py              # all sizes
    python sweep_crop_with_rotation.py --sizes small medium
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.cropping.train import train

LOG_PATH = Path("logs/sweep_crop_angle.jsonl")

CONFIGS = {
    "small": {
        "backbone":        "efficientnet_b0",
        "epochs":          25,
        "batch_size":      32,
        "lr":              1e-4,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
        "use_angle_head":  True,
        "ckpt_tag":        "",
    },
    "small_lowlr": {
        "backbone":        "efficientnet_b0",
        "epochs":          25,
        "batch_size":      32,
        "lr":              1e-5,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
        "use_angle_head":  True,
        "ckpt_tag":        "_lr1e5",
    },
    "medium": {
        "backbone":        "efficientnet_b3",
        "epochs":          25,
        "batch_size":      16,
        "lr":              1e-4,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
        "use_angle_head":  True,
        "ckpt_tag":        "",
    },
    "medium_lowlr": {
        "backbone":        "efficientnet_b3",
        "epochs":          25,
        "batch_size":      16,
        "lr":              1e-5,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
        "use_angle_head":  True,
        "ckpt_tag":        "_lr1e5",
    },
    "large": {
        "backbone":        "vit_base_patch14_reg4_dinov2",
        "epochs":          40,
        "batch_size":      8,
        "lr":              1e-5,
        "warmup_epochs":   3,
        "grad_checkpoint": True,
        "use_angle_head":  True,
        "ckpt_tag":        "",
    },
    "resnet50": {
        "backbone":        "resnet50",
        "epochs":          25,
        "batch_size":      16,
        "lr":              1e-4,
        "warmup_epochs":   0,
        "grad_checkpoint": False,
        "use_angle_head":  True,
        "ckpt_tag":        "",
    },
    "siglip": {
        "backbone":        "vit_base_patch16_siglip_512",
        "epochs":          40,
        "batch_size":      8,
        "lr":              1e-5,
        "warmup_epochs":   3,
        "grad_checkpoint": True,
        "use_angle_head":  True,
        "ckpt_tag":        "",
    },
    "large_dinov2": {
        "backbone":        "vit_large_patch14_dinov2",
        "epochs":          40,
        "batch_size":      4,
        "lr":              1e-5,
        "warmup_epochs":   3,
        "grad_checkpoint": True,
        "use_angle_head":  True,
        "ckpt_tag":        "",
    },
    "large_dinov2_lr1e6": {
        "backbone":        "vit_large_patch14_dinov2",
        "epochs":          60,
        "batch_size":      4,
        "lr":              1e-6,
        "warmup_epochs":   5,
        "grad_checkpoint": True,
        "use_angle_head":  True,
        "ckpt_tag":        "_lr1e6",
    },
}

_DEFAULT_ORDER = ["small", "small_lowlr", "medium", "medium_lowlr", "large"]
_EXTRA_ORDER   = ["resnet50", "siglip", "large_dinov2"]


def main(sizes: list[str], use_player_bbox: bool = False,
         resume: bool = False) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not resume:
        LOG_PATH.unlink(missing_ok=True)

    results: list[dict] = []
    for size in sizes:
        cfg = dict(CONFIGS[size])
        if use_player_bbox:
            cfg["use_player_bbox"] = True
            cfg["ckpt_tag"] = cfg.get("ckpt_tag", "") + "_pb"
        if resume:
            cfg["resume"] = True
        print(f"\n{'='*65}")
        pb_str = "  +player-bbox" if use_player_bbox else ""
        print(f"  {size.upper()} (box+angle{pb_str}): {cfg['backbone']}  lr={cfg['lr']:.0e}")
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
    print(f"\n{'='*95}")
    print(f"  {'Size':<14} {'Backbone':<38} {'lr':>7} {'Mean IoU':>9} {'Median':>8} "
          f"{'>0.7':>6} {'>0.8':>6} {'Angle MAE':>10}")
    print("-" * 95)
    best = max(results, key=lambda x: x.get("mean_iou", 0))
    for r in sorted(results, key=lambda x: x.get("mean_iou", 0), reverse=True):
        tm   = r.get("test_metrics", r)
        m_   = tm.get("mean_iou",       r.get("mean_iou",   0))
        med  = tm.get("median_iou",     r.get("median_iou", 0))
        g70  = tm.get("iou_gt70",       0)
        g80  = tm.get("iou_gt80",       0)
        mae  = tm.get("angle_mae_deg",  None)
        cfg  = CONFIGS.get(r.get("size", ""), {})
        lr_v = cfg.get("lr", 0)
        mae_str = f"{mae:>9.2f}°" if mae is not None else "       N/A"
        mark = "  <--" if r is best else ""
        print(f"  {r.get('size','?'):<14} {r['backbone']:<38} {lr_v:>7.0e} "
              f"{m_:>9.4f} {med:>8.4f} {g70:>5.1%} {g80:>5.1%} {mae_str}{mark}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+",
        choices=list(CONFIGS.keys()),
        default=_DEFAULT_ORDER,
    )
    parser.add_argument(
        "--extra", action="store_true",
        help="Run the extra 3 backbones (resnet50, siglip, large_dinov2)",
    )
    parser.add_argument(
        "--player-bbox", action="store_true",
        help="Condition model on YOLO player union bbox (requires data/player_bboxes.json). "
             "Checkpoints tagged with _pb suffix.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from existing checkpoints (skips completed backbones).",
    )
    args = parser.parse_args()
    sizes = _EXTRA_ORDER if args.extra else args.sizes
    main(sizes, use_player_bbox=args.player_bbox, resume=args.resume)
