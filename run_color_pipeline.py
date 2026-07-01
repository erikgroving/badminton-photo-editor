"""Auto-chain: judge training → color correction training (param model)."""
import subprocess, sys

steps = [
    ("Judge training",
     [sys.executable, "-m", "models.judge.train"]),
    ("Color correction (param model + GAN judge)",
     [sys.executable, "-m", "models.color_correction.train", "--model", "param"]),
]

for name, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print("="*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {name} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"DONE: {name}")
