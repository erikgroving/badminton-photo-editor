"""
Experimental sweep of large / modern-pretrained backbones.

Models:
  1. vit_large_patch14_dinov2     — DINOv2-Large  (304M, 518×518, self-supervised ViT)
  2. vit_base_patch14_reg4_dinov2 — DINOv2-Base+  ( 87M, 518×518, with register tokens)
  3. vit_base_patch16_siglip_512  — SigLIP-512    ( 94M, 512×512, CLIP-style pretraining)

All three use gradient checkpointing to stay within VRAM.
Checkpoints saved as culling_<backbone>.pt (standard naming, picked up by recall_vs_selection.py).

Usage:
    python sweep_large.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_BATCH_SIZE, CULL_EPOCHS, CULL_FN_WEIGHT

# Large ViTs need a much smaller LR than EfficientNet/ResNet — 1e-4 causes collapse
LARGE_LR = 1e-5
from models.culling.train import train

# (backbone, batch_size, notes)
EPOCHS = 8

LARGE_MODELS = [
    # SigLIP first — native 512, smallest of the three, good warmup
    ("vit_base_patch16_siglip_512",  4,  "SigLIP 512 — CLIP-style, native resolution"),
    # DINOv2-Base with register tokens — good middle ground
    ("vit_base_patch14_reg4_dinov2", 4,  "DINOv2 Base+reg — 518×518, self-supervised"),
    # DINOv2-Large — the big one; batch=1 to stay within VRAM
    ("vit_large_patch14_dinov2",     1,  "DINOv2 Large — 304M params, batch=1"),
]


def main():
    results = []
    for backbone, batch_size, note in LARGE_MODELS:
        ckpt = CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt"
        print(f"\n{'='*72}")
        print(f"  {backbone}")
        print(f"  {note}")
        print(f"  batch={batch_size}  epochs={EPOCHS}  fn_weight={CULL_FN_WEIGHT}")
        print("=" * 72)

        if ckpt.exists():
            print("  [SKIP] checkpoint already exists")
            import torch
            try:
                ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
                m  = dict(ck.get("metrics", {}), backbone=backbone)
                results.append(m)
            except Exception:
                pass
            continue

        try:
            m = train(
                backbone=backbone,
                epochs=EPOCHS,
                batch_size=batch_size,
                fn_weight=CULL_FN_WEIGHT,
                lr=LARGE_LR,
                grad_checkpoint=True,
                warmup_epochs=2,
            )
            results.append(m)
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append({"backbone": backbone, "recall": 0.0,
                            "selection_rate": 0.0, "asym_cost": float("inf")})

    print("\n" + "=" * 72)
    print(f"  {'Backbone':<42} {'Recall':>8} {'Sel%':>8} {'Cost':>10}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r.get('backbone','?'):<42}"
              f" {r.get('recall',0):>8.1%}"
              f" {r.get('selection_rate',0):>8.1%}"
              f" {r.get('asym_cost',0):>10.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
