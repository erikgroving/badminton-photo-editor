"""
Run all 5 crop research experiments sequentially (one GPU process at a time).
This avoids CUDA memory contention that caused crashes when running in parallel.

After all experiments complete, runs the DINOv2 _pb sweep --resume for ep31-40.

Usage:
    python run_experiments_sequential.py
    python run_experiments_sequential.py --skip-dinov2
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = sys.executable

EXPERIMENTS = [
    ("exp1", ROOT / "experiments" / "exp1_rich_features.py"),
    ("exp2", ROOT / "experiments" / "exp2_rule_of_thirds.py"),
    ("exp3", ROOT / "experiments" / "exp3_rule_of_thirds.py"),
    ("exp4", ROOT / "experiments" / "exp4_spatial_features.py"),
    ("exp5", ROOT / "experiments" / "exp5_image_stats.py"),
]

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def run_experiment(name: str, script: Path) -> bool:
    """Run one experiment script; return True on success."""
    log_path = LOG_DIR / f"seq_{name}_runner.log"
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Starting {name}: {script.name}")
    print(f"  Log: {log_path}")
    print("=" * 70)

    with open(log_path, "w") as fout:
        proc = subprocess.Popen(
            [PYTHON, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fout.write(line)
        proc.wait()

    elapsed = (time.time() - t0) / 60
    if proc.returncode == 0:
        print(f"\n  [{name}] Finished in {elapsed:.1f} min  (exit 0)")
        return True
    else:
        print(f"\n  [{name}] FAILED (exit {proc.returncode}) after {elapsed:.1f} min")
        return False


def run_dinov2_resume() -> bool:
    """Resume DINOv2 _pb training for remaining epochs."""
    log_path = LOG_DIR / "seq_dinov2_resume.log"
    script   = ROOT / "sweep_crop_with_rotation.py"
    t0       = time.time()
    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Resuming DINOv2 _pb (ep31-40)")
    print(f"  Log: {log_path}")
    print("=" * 70)

    with open(log_path, "w") as fout:
        proc = subprocess.Popen(
            [PYTHON, str(script), "--sizes", "large", "--player-bbox", "--resume"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fout.write(line)
        proc.wait()

    elapsed = (time.time() - t0) / 60
    if proc.returncode == 0:
        print(f"\n  [DINOv2] Finished in {elapsed:.1f} min  (exit 0)")
        return True
    else:
        print(f"\n  [DINOv2] FAILED (exit {proc.returncode}) after {elapsed:.1f} min")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dinov2", action="store_true",
                        help="Skip the DINOv2 resume after experiments")
    parser.add_argument("--only", nargs="+",
                        choices=["exp1", "exp2", "exp3", "exp4", "exp5", "dinov2"],
                        help="Only run specified experiments")
    args = parser.parse_args()

    results = {}
    for name, script in EXPERIMENTS:
        if args.only and name not in args.only:
            print(f"  Skipping {name} (not in --only)")
            continue
        ok = run_experiment(name, script)
        results[name] = "OK" if ok else "FAILED"

    if not args.skip_dinov2 and (not args.only or "dinov2" in args.only):
        ok = run_dinov2_resume()
        results["dinov2"] = "OK" if ok else "FAILED"

    print(f"\n{'='*70}")
    print("  Results summary:")
    for k, v in results.items():
        mark = "✓" if v == "OK" else "✗"
        print(f"    {mark} {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
