"""
Run cache_pose_keypoints until all 3698 entries are processed.
Runs in a tight loop since each run may be killed partway through.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TARGET = 3698

while True:
    cache_file = ROOT / "data" / "pose_keypoints.json"
    if cache_file.exists():
        cache = json.load(open(cache_file))
        n_done = sum(1 for v in cache.values() if v is not None or v is None)
        n_done = len(cache)
    else:
        n_done = 0

    print(f"Cache has {n_done}/{TARGET} entries. Remaining: {TARGET - n_done}")
    if n_done >= TARGET:
        print("All entries cached!")
        break

    result = subprocess.run(
        [sys.executable, "data/cache_pose_keypoints.py"],
        cwd=str(ROOT),
        capture_output=False,
    )
    print(f"Run completed with exit code {result.returncode}")
