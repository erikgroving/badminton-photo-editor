"""
Run color conditioning experiments c1-c6 sequentially on one GPU.

Usage:
    python run_color_experiments.py
    python run_color_experiments.py --only color_exp2_camera_wb color_exp6_combined
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = sys.executable
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("color_exp1_exif",        ROOT / "experiments" / "color_exp1_exif.py"),
    ("color_exp2_camera_wb",   ROOT / "experiments" / "color_exp2_camera_wb.py"),
    ("color_exp3_classic_wb",  ROOT / "experiments" / "color_exp3_classic_wb.py"),
    ("color_exp4_region_stats",ROOT / "experiments" / "color_exp4_region_stats.py"),
    ("color_exp5_histogram",   ROOT / "experiments" / "color_exp5_histogram.py"),
    ("color_exp6_combined",    ROOT / "experiments" / "color_exp6_combined.py"),
]


def run_one(name: str, script: Path) -> bool:
    log_path = LOG_DIR / f"{name}.log"
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name}")
    print(f"  Log: {log_path}")
    print("=" * 70)

    with open(log_path, "w", encoding="utf-8", errors="replace") as fout:
        proc = subprocess.Popen(
            [PYTHON, str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
        )
        assert proc.stdout
        for line in proc.stdout:
            try:
                sys.stdout.write(line); sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                sys.stdout.flush()
            fout.write(line)
        proc.wait()

    elapsed = (time.time() - t0) / 60
    ok = proc.returncode == 0
    print(f"\n  [{name}] {'OK' if ok else 'FAILED'}  ({elapsed:.1f} min)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+",
                        choices=[n for n, _ in EXPERIMENTS],
                        help="Run only specified experiments")
    args = parser.parse_args()

    results = {}
    for name, script in EXPERIMENTS:
        if args.only and name not in args.only:
            print(f"  Skipping {name}")
            continue
        results[name] = "OK" if run_one(name, script) else "FAILED"

    print(f"\n{'='*70}")
    print("  Color experiment results:")
    for k, v in results.items():
        print(f"    {'OK' if v == 'OK' else 'XX'} {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
