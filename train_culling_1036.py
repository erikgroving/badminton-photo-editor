"""
Train DINOv2-Large culling classifier at 1036×1036 (= 14×74, native patch grid).

Images are loaded directly from the 6000×4000 embedded JPEG in each CR3 and
downsampled to 1036, giving real sensor detail vs. the 512-px thumbnail path.

Position embeddings are interpolated from the pretrained 518-px grid via
dynamic_img_size=True — no extra pre-training needed.

Memory is significant (~8-12 GB VRAM); gradient checkpointing is required.
Batch size 1 is recommended unless you have ≥24 GB VRAM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_FN_WEIGHT
from models.culling.train import train

BACKBONE    = "vit_large_patch14_dinov2"
INPUT_SIZE  = 1036   # 14 × 74 — divisible by patch_size=14
BATCH       = 1
EPOCHS      = 8
LR          = 3e-6
FN_WEIGHT   = CULL_FN_WEIGHT
WARMUP      = 2

ckpt = CHECKPOINTS_DIR / f"culling_{BACKBONE.replace('/', '_')}_1036.pt"

if __name__ == "__main__":
    print(f"\n{'='*72}")
    print(f"  {BACKBONE} @ {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"  batch={BATCH}  lr={LR}  epochs={EPOCHS}  fn_weight={FN_WEIGHT}  warmup={WARMUP}")
    print(f"  dynamic_img_size=True  (position embeddings interpolated from 518-px grid)")
    print("=" * 72)

    if ckpt.exists():
        print("  [SKIP] checkpoint already exists — delete it to retrain")
        print(f"  {ckpt}")
    else:
        train(
            backbone=BACKBONE,
            epochs=EPOCHS,
            batch_size=BATCH,
            fn_weight=FN_WEIGHT,
            lr=LR,
            grad_checkpoint=True,
            warmup_epochs=WARMUP,
            force_input_size=INPUT_SIZE,
            dynamic_img_size=True,
            ckpt_suffix="_1036",
        )
        print(f"\nDone. Checkpoint: {ckpt}")
