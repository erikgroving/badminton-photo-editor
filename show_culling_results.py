import torch
from pathlib import Path

ckpt_dir = Path("checkpoints")
pts = sorted(ckpt_dir.glob("culling_*.pt"), key=lambda f: f.stat().st_mtime)
if not pts:
    print("No culling checkpoints found yet.")
else:
    for pt in pts:
        ck = torch.load(str(pt), map_location="cpu", weights_only=False)
        m = ck.get("metrics", {})
        print(pt.name)
        print(f"  best epoch : {ck.get('epoch', '?')}")
        print(f"  recall     : {m.get('recall', 0):.1%}  (% of kept photos selected)")
        print(f"  selection  : {m.get('selection_rate', 0):.1%}  (% of ALL photos flagged keep)")
        print(f"  F2 score   : {m.get('f2', 0):.4f}")
        print(f"  asym cost  : {m.get('asym_cost', 0):.1f}")
        print()
