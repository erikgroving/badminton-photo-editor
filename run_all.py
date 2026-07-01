"""
Quickstart: full pipeline from raw training data → trained models → UI.

Run each step in order. Each step is idempotent (skips if already done).
You can run individual steps by passing their number, e.g.:
    python run_all.py 2      # just run step 2
    python run_all.py 2 3 4  # run steps 2, 3, 4
    python run_all.py        # print this menu and exit
"""
import subprocess
import sys

STEPS = [
    (1,  "Build raw→edited mapping (fast, filename-based)",
         "python -m data.mapping --rebuild --verify --sanity-check 10"),

    (2,  "Extract crop ground-truth from paired images (slow, runs once)",
         "python -m data.crop_detector --workers 4"),

    (3,  "Train culling model — default backbone only",
         "python -m models.culling.train"),

    (4,  "Sweep all culling backbones and compare",
         "python -m models.culling.sweep"),

    (5,  "Train crop model — default backbone only",
         "python -m models.cropping.train"),

    (6,  "Sweep all crop backbones and compare",
         "python -m models.cropping.sweep"),

    (7,  "Train judge discriminator — default backbone",
         "python -m models.judge.train"),

    (8,  "Sweep all judge backbones and compare",
         "python -m models.judge.sweep"),

    (9,  "Extract color correction ground-truth (slow, runs once)",
         "python -m data.color_analyzer --workers 4"),

    (10, "Train color correction models (param + U-Net) with GAN judge",
         "python -m models.color_correction.train --model both"),

    (11, "Launch the interactive review UI",
         "python -m ui.app"),
]

def main():
    requested = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else []

    if not requested:
        print("\nBadminton Photo Editor — training pipeline\n")
        for num, desc, cmd in STEPS:
            print(f"  Step {num:2d}: {desc}")
            print(f"           {cmd}\n")
        print("Usage: python run_all.py [step numbers...]")
        print("       python run_all.py 1 2 3   # run steps 1, 2, 3 in order")
        return

    for num, desc, cmd in STEPS:
        if num in requested:
            print(f"\n{'='*60}")
            print(f"Step {num}: {desc}")
            print(f"  > {cmd}")
            print("=" * 60)
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"\nStep {num} failed with exit code {result.returncode}. Stopping.")
                sys.exit(result.returncode)

if __name__ == "__main__":
    main()
