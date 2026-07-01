import torch
from pathlib import Path
for f in sorted(Path('checkpoints').glob('color_param_*.pt')):
    d = torch.load(f, map_location='cpu', weights_only=False)
    print(f"{f.name}")
    print(f"  keys: {list(d.keys())}")
    print(f"  metrics: {d.get('metrics', d.get('val_loss', '?'))}")
    print()
