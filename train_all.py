"""
Overnight training orchestration.
Runs every step in dependency order, logging each step's output to
logs/<step_name>.log so you can review results in the morning.

Usage:
    python train_all.py
    python train_all.py --resume   # skip steps whose output already exists
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout so Unicode in subprocess output doesn't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

STEPS = [
    {
        "name":    "mapping",
        "cmd":     "python -m data.mapping --rebuild --verify --sanity-check 10",
        "creates": "data/mapping.json",
        "desc":    "Build raw→edited filename mapping + save 10 sanity-check image pairs",
    },
    {
        "name":    "watermark_detect",
        "cmd":     "python -m data.detect_watermark --samples 200 --apply",
        "creates": "sanity_check/watermark_heatmap.jpg",
        "desc":    "Auto-detect watermark region and write to config.py",
    },
    {
        "name":    "culling_sweep",
        "cmd":     "python -m models.culling.sweep",
        "creates": "checkpoints/culling_mobilenetv3_large_100.pt",
        "desc":    "Train all culling backbones and compare",
    },
    {
        "name":    "culling_fn_weight_sweep",
        "cmd":     "python -m models.culling.fn_weight_sweep --backbone efficientnet_b0",
        "creates": "checkpoints/culling_efficientnet_b0_w2.pt",
        "desc":    "Sweep FN penalty weights [2,5,10,15,20,30] to show recall/selection trade-off",
    },
    {
        "name":    "culling_attribute_train",
        "cmd":     "python -m models.culling.attribute_train --backbone efficientnet_b0",
        "creates": "checkpoints/culling_attr_efficientnet_b0_w15.pt",
        "desc":    "Train attribute-aware culling model (CNN + blur/exposure/face features)",
    },
    {
        "name":    "culling_cluster",
        "cmd":     "python -m models.culling.cluster_culling --backbone efficientnet_b0 --k 6",
        "creates": "checkpoints/culling_cluster_k6_efficientnet_b0.pkl",
        "desc":    "Unsupervised clustering experiment: K-means + per-cluster specialists + meta-learner",
    },
    {
        "name":    "crop_detector",
        "cmd":     "python -m data.crop_detector --workers 4",
        "creates": "data/crop_gt.json",
        "desc":    "Extract crop ground-truth from paired images (slow CPU task)",
    },
    {
        "name":    "judge_sweep",
        "cmd":     "python -m models.judge.sweep",
        "creates": "checkpoints/judge_efficientnet_b0.pt",
        "desc":    "Train all judge/discriminator backbones and compare",
    },
    {
        "name":    "crop_sweep",
        "cmd":     "python -m models.cropping.sweep",
        "creates": "checkpoints/cropping_resnet50.pt",
        "desc":    "Train all crop backbones and compare",
    },
    {
        "name":    "color_gt",
        "cmd":     "python -m data.xmp_reader --out data/color_gt.json",
        "creates": "data/color_gt.json",
        "desc":    "Extract color GT from XMP metadata embedded in edited JPEGs (fast, exact LR values)",
    },
    {
        "name":    "color_training",
        "cmd":     "python -m models.color_correction.train --model both",
        "creates": "checkpoints/color_param.pt",
        "desc":    "Train param model + U-Net with GAN judge (alternating training)",
    },
]


def run_step(step: dict, resume: bool) -> bool:
    creates = Path(step["creates"])
    if resume and creates.exists():
        print(f"  [OK] Skipping '{step['name']}' (output already exists: {creates})")
        return True

    log_file = LOG_DIR / f"{step['name']}.log"
    print(f"\n{'='*65}")
    print(f"  Step: {step['name']}")
    print(f"  {step['desc']}")
    print(f"  Command: {step['cmd']}")
    print(f"  Log: {log_file}")
    print("=" * 65)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # force line-by-line flushing from subprocesses

    start = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            step["cmd"], shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
        proc.wait()

    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"\n  [FAIL] FAILED (exit {proc.returncode}) after {elapsed/60:.1f} min")
        print(f"    Full log: {log_file}")
        return False

    print(f"\n  [OK] Done in {elapsed/60:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps whose output file already exists")
    args = parser.parse_args()

    print("=" * 65)
    print("  Badminton Photo Editor — Overnight Training Pipeline")
    print("=" * 65)
    print(f"  Steps: {len(STEPS)}")
    print(f"  Logs:  {LOG_DIR.absolute()}")
    print(f"  Resume mode: {args.resume}")
    print()

    overall_start = time.time()
    for step in STEPS:
        ok = run_step(step, args.resume)
        if not ok:
            print(f"\nPipeline stopped at step '{step['name']}'. Check {LOG_DIR}/{step['name']}.log")
            sys.exit(1)

    total = (time.time() - overall_start) / 60
    print(f"\n{'='*65}")
    print(f"  All steps complete in {total:.1f} min.")
    print(f"  Results summary:")
    print(f"    Sanity-check pairs : sanity_check/pair_*.jpg")
    print(f"    Watermark heatmap  : sanity_check/watermark_heatmap.jpg")
    print(f"    Culling logs       : logs/culling_sweep.log")
    print(f"    Crop logs          : logs/crop_sweep.log")
    print(f"    Judge logs         : logs/judge_sweep.log")
    print(f"    Color logs         : logs/color_training.log")
    print(f"    Checkpoints        : checkpoints/")
    print(f"\n  Launch the UI:  python -m ui.app")
    print("=" * 65)


if __name__ == "__main__":
    main()
