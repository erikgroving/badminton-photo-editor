"""Verify 0.65 threshold covers all 5 known player groups."""
import re
from pathlib import Path
import numpy as np
import torch
from inference.player_coverage import (
    build_detector, build_embedder, detect_player_crops,
    embed_crops, cluster_embeddings,
)
from data.raw_reader import extract_thumbnail

GROUPS = {
    1: set(range(1316, 1344)),
    2: set(range(1344, 1390)),
    3: set(range(1390, 1447)),
    4: set(range(1447, 1480)),
    5: set(range(1480, 1487)),
}

def num(fname):
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else 0

def group(fname):
    n = num(fname)
    for g, rng in GROUPS.items():
        if n in rng: return g
    return 0

demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')
_seen = set()
raws = []
for p in sorted(demo_dir.glob('*')):
    if p.suffix.lower() == '.cr3' and p.name.lower() not in _seen:
        _seen.add(p.name.lower())
        raws.append(p)
print(f'Unique photos: {len(raws)}  (sample nums: {[num(p.name) for p in raws[:5]]})')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = build_detector()
embedder = build_embedder(device)

all_embs, all_fnames = [], []
for p in raws:
    img = extract_thumbnail(p, size=(448, 448))
    crops = detect_player_crops(img, detector)
    if crops:
        embs = embed_crops(crops, embedder, device)
        for row in embs:
            all_embs.append(row)
            all_fnames.append(p.name)

print(f'Total crops: {len(all_embs)}\n')
emb_matrix = np.stack(all_embs)

for thresh in [0.60, 0.65, 0.70]:
    labels = cluster_embeddings(emb_matrix, thresh)
    n_clusters = int(labels.max()) + 1

    cluster_to_groups: dict[int, set] = {}
    for row_idx, cluster_id in enumerate(labels):
        g = group(all_fnames[row_idx])
        cluster_to_groups.setdefault(int(cluster_id), set()).add(g)

    covered = set()
    for gs in cluster_to_groups.values():
        covered |= gs
    covered.discard(0)

    print(f'Threshold {thresh:.2f}: {n_clusters} clusters, covered groups: {sorted(covered)}')
    if {1,2,3,4,5}.issubset(covered):
        print('  All 5 players covered')
    else:
        print(f'  Missing: {sorted({1,2,3,4,5} - covered)}')

    # Show per-cluster composition
    for cid in sorted(cluster_to_groups):
        gs = sorted(cluster_to_groups[cid])
        print(f'    cluster {cid}: groups {gs}')
    print()
