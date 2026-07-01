"""
Retrain mobilenetv3_large_100 on all three split strategies so the comparison
in recall_vs_selection.py has no data leakage.

Run AFTER the main backbone sweep has finished (sweep saves b0/b3/resnet50).

Usage:
    python retrain_mobilenetv3_splits.py
"""
import subprocess
import sys
import time
from pathlib import Path

BACKBONE = "mobilenetv3_large_100"

RUNS = [
    {
        "label":       "event-split",
        "mapping":     "data/mapping_event.json",
        "ckpt_suffix": "_event",
    },
    {
        "label":       "burst-60",
        "mapping":     "data/mapping_burst60.json",
        "ckpt_suffix": "_burst60",
    },
    {
        "label":       "random-split",
        "mapping":     "data/mapping_random.json",
        "ckpt_suffix": "_random",
    },
    # burst-1sec is in the main checkpoint (no suffix): culling_mobilenetv3_large_100.pt
]

def run(label: str, mapping: str, ckpt_suffix: str) -> None:
    cmd = (
        f"python -m models.culling.train "
        f"--backbone {BACKBONE} "
        f"--mapping {mapping} "
        f"--ckpt-suffix {ckpt_suffix}"
    )
    print(f"\n{'='*65}")
    print(f"  {label}  →  culling_{BACKBONE}{ckpt_suffix}.pt")
    print(f"  {cmd}")
    print("=" * 65)
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = (time.time() - start) / 60
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.1f} min")
        sys.exit(1)
    print(f"  Done in {elapsed:.1f} min")


if __name__ == "__main__":
    for run_cfg in RUNS:
        run(**run_cfg)
    print("\nAll mobilenetv3 split variants trained.")
    print("Run: python recall_vs_selection.py")
