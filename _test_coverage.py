import json
from pathlib import Path
import torch

demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')
raws = sorted(demo_dir.glob('*.CR3')) + sorted(demo_dir.glob('*.cr3'))
photos = [{'filename': p.name, 'score': 0.1, 'decision': 'culled', 'rank': i+1}
          for i, p in enumerate(raws)]
print(f'Photos: {len(photos)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from inference.player_coverage import run_coverage_guarantee

def cb(frac, msg):
    print(f'  [{frac:.0%}] {msg}')

updated, stats = run_coverage_guarantee(photos, demo_dir, (448, 448), device, progress_cb=cb)
print()
print('Stats:', json.dumps(stats, indent=2))
promoted = [p for p in updated if p.get('coverage_promoted')]
print(f'Promoted: {len(promoted)} photos across {stats["n_clusters"]} clusters')
for p in promoted:
    print(f'  {p["filename"]}')
