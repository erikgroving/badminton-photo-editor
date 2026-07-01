import torch
from pathlib import Path

print("=== CROP ===")
for f in sorted(Path("checkpoints").glob("cropping_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    m = d.get("metrics", {})
    print(f"  {f.name}  median_iou={m.get('median_iou',0):.4f}  mean_iou={m.get('mean_iou',0):.4f}")

print()
print("=== COLOR ===")
for f in sorted(Path("checkpoints").glob("color_param_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    m = d.get("metrics", {})
    print(f"  {f.name}  val_l1={m.get('val_l1',9):.4f}  judge={m.get('judge_score',0):.3f}  player_crop={d.get('player_crop',False)}")
