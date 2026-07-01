"""
Retrain DINOv2-Large with a lower LR.

batch=1 means 26k gradient steps/epoch — the effective update magnitude is
4x larger than batch=4, so we scale LR down proportionally: 1e-5 / 4 ≈ 3e-6.
Still uses warmup (2 epochs) + cosine decay + gradient checkpointing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_FN_WEIGHT
from models.culling.train import train

BACKBONE   = "vit_large_patch14_dinov2"
BATCH      = 1
EPOCHS     = 8
LR         = 3e-6
FN_WEIGHT  = CULL_FN_WEIGHT
WARMUP     = 2

ckpt = CHECKPOINTS_DIR / f"culling_{BACKBONE.replace('/', '_')}.pt"

if __name__ == "__main__":
    print(f"\n{'='*72}")
    print(f"  {BACKBONE}")
    print(f"  DINOv2 Large — 304M params, batch={BATCH}, lr={LR}")
    print(f"  epochs={EPOCHS}  fn_weight={FN_WEIGHT}  warmup={WARMUP}")
    print("=" * 72)

    if ckpt.exists():
        print("  [SKIP] checkpoint already exists — delete it first to retrain")
    else:
        train(
            backbone=BACKBONE,
            epochs=EPOCHS,
            batch_size=BATCH,
            fn_weight=FN_WEIGHT,
            lr=LR,
            grad_checkpoint=True,
            warmup_epochs=WARMUP,
        )
        print(f"\nDone. Checkpoint: {ckpt}")
