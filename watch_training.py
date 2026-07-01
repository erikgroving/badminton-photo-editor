"""
Training status dashboard.

Usage:
    python watch_training.py              # print status once
    python watch_training.py --watch      # refresh every 30 s
    python watch_training.py --watch 10   # refresh every 10 s
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_BACKBONE_CANDIDATES

_LOGS_DIR = Path("logs")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _gpu_util() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        ).strip()
        gpu_pct, mem_pct, mem_used, mem_total = out.split(", ")
        return f"GPU {gpu_pct}%  VRAM {int(mem_used)/1024:.1f}/{int(mem_total)/1024:.1f} GB"
    except Exception:
        return "GPU info unavailable"


def _load_progress(backbone: str) -> list[dict]:
    """Read all epoch lines from the progress JSONL for this backbone."""
    safe = backbone.replace("/", "_")
    p = _LOGS_DIR / f"progress_culling_{safe}.jsonl"
    if not p.exists():
        return []
    lines = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lines


def _ckpt_metrics(backbone: str) -> dict | None:
    """Load saved metrics from the best checkpoint file."""
    safe = backbone.replace("/", "_")
    pt = CHECKPOINTS_DIR / f"culling_{safe}.pt"
    if not pt.exists():
        return None
    try:
        import torch
        ck = torch.load(str(pt), map_location="cpu", weights_only=False)
        m = ck.get("metrics", {})
        m["_epoch"]  = ck.get("epoch")
        m["_mtime"]  = pt.stat().st_mtime
        return m
    except Exception:
        return None


def _ckpt_duration(backbone: str, prev_backbone: str | None) -> str:
    """Estimate duration from checkpoint mtime delta."""
    safe = backbone.replace("/", "_")
    pt   = CHECKPOINTS_DIR / f"culling_{safe}.pt"
    if not pt.exists():
        return "?"
    end_t = pt.stat().st_mtime
    if prev_backbone:
        prev_safe = prev_backbone.replace("/", "_")
        prev_pt   = CHECKPOINTS_DIR / f"culling_{prev_safe}.pt"
        if prev_pt.exists():
            return _fmt_duration(end_t - prev_pt.stat().st_mtime)
    # First backbone: try to infer start from progress file or log ctime
    log = Path("logs") / "culling_sweep.log"
    if log.exists() and log.stat().st_size == 0:
        # Log was just created at sweep start — use its mtime as start
        return _fmt_duration(end_t - log.stat().st_mtime)
    return "?"


def print_status() -> None:
    now = time.time()
    print()
    print("=" * 70)
    print(f"  Culling Sweep - Training Status    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {_gpu_util()}")
    print("=" * 70)

    completed   = []
    in_progress = None
    queued      = []

    for backbone in CULL_BACKBONE_CANDIDATES:
        ckpt = _ckpt_metrics(backbone)
        progress = _load_progress(backbone)
        if ckpt is not None and (not progress or progress[-1]["epoch"] == progress[-1]["total"]):
            completed.append((backbone, ckpt))
        elif progress:
            in_progress = (backbone, progress)
        elif ckpt is None and in_progress is None and not queued:
            # No checkpoint, no progress, nothing in-progress yet — this is next up
            in_progress = (backbone, [])
        else:
            queued.append(backbone)

    # ── Completed ──────────────────────────────────────────────────────────
    if completed:
        print(f"\n  Completed ({len(completed)}/{len(CULL_BACKBONE_CANDIDATES)} backbones):")
        prev = None
        for backbone, m in completed:
            beta_key = next((k for k in m if k.startswith("f") and k[1:].isdigit()), "f2")
            dur = _ckpt_duration(backbone, prev)
            print(
                f"    {backbone:<30}  "
                f"recall={m.get('recall', 0):.3f}  "
                f"sel={m.get('selection_rate', 0):.1%}  "
                f"{beta_key}={m.get(beta_key, 0):.4f}  "
                f"cost={m.get('asym_cost', 0):.0f}  "
                f"[{dur}]"
            )
            prev = backbone

    # ── In-progress ────────────────────────────────────────────────────────
    if in_progress:
        backbone, rows = in_progress
        print(f"\n  Currently training: {backbone}")
        if rows:
            latest  = rows[-1]
            epoch   = latest["epoch"]
            total   = latest["total"]
            elapsed = latest.get("elapsed_s", 0)
            epoch_s = latest.get("epoch_s", 0)

            # Average epoch time from last 3 epochs (stabilises after warmup)
            recent    = rows[-3:] if len(rows) >= 3 else rows
            avg_epoch = sum(r.get("epoch_s", epoch_s) for r in recent) / len(recent)
            remaining = (total - epoch) * avg_epoch
            eta_str   = _fmt_duration(remaining)

            beta_key = next((k for k in latest if k.startswith("f") and k[1:].isdigit()), "f2")
            print(
                f"    Epoch {epoch}/{total}  |  "
                f"loss={latest.get('loss', 0):.4f}  |  "
                f"elapsed {_fmt_duration(elapsed)}"
            )
            print(
                f"    Val: recall={latest.get('recall', 0):.3f}  "
                f"sel={latest.get('selection_rate', 0):.1%}  "
                f"{beta_key}={latest.get(beta_key, 0):.4f}  "
                f"cost={latest.get('asym_cost', 0):.0f}"
            )
            print(
                f"    Avg epoch: {_fmt_duration(avg_epoch)}  |  "
                f"ETA this backbone: ~{eta_str}"
            )

            # ETA for full sweep (queued backbones assumed same pace)
            if queued:
                remaining_total = remaining + len(queued) * total * avg_epoch
                print(f"    ETA full sweep:  ~{_fmt_duration(remaining_total)}")
        else:
            # No progress file — backbone hasn't started its first epoch yet
            safe = backbone.replace("/", "_")
            pt   = CHECKPOINTS_DIR / f"culling_{safe}.pt"
            if not pt.exists():
                print(f"    Waiting for first epoch... (no progress data yet)")
            else:
                print(f"    Checkpoint exists but no progress file — may be running from prior session")

    # ── Queued ─────────────────────────────────────────────────────────────
    if queued:
        print(f"\n  Queued: {', '.join(queued)}")

    # ── Nothing running ────────────────────────────────────────────────────
    if not in_progress and not completed:
        print("\n  No training activity detected.")
        print("  Run:  python train_all.py --resume")

    # ── Best so far ────────────────────────────────────────────────────────
    if len(completed) > 1:
        best = min(completed, key=lambda x: x[1].get("asym_cost", float("inf")))
        print(f"\n  Best so far: {best[0]}  (asym_cost={best[1].get('asym_cost', 0):.0f})")

    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", nargs="?", const=30, type=int, metavar="SECS",
                        help="Refresh every N seconds (default 30)")
    args = parser.parse_args()

    if args.watch is None:
        print_status()
    else:
        interval = args.watch
        print(f"Watching training (refresh every {interval}s) — Ctrl+C to stop")
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print_status()
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break


if __name__ == "__main__":
    main()
