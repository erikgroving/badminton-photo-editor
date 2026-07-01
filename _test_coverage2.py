"""Sweep distance thresholds to find the right one for 5-player scenario."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

demo_dir = Path(r'C:\Users\erikg\Downloads\Demo')
# Deduplicate — Windows glob is case-insensitive so *.CR3 and *.cr3 both match
_seen = set()
raws = []
for p in sorted(demo_dir.glob('*.CR3')) + sorted(demo_dir.glob('*.cr3')):
    if p.name.lower() not in _seen:
        _seen.add(p.name.lower())
        raws.append(p)
print(f'Unique photos: {len(raws)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from inference.player_coverage import build_detector, build_embedder, detect_player_crops, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

detector = build_detector()
embedder = build_embedder(device)

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print('Detecting + embedding...')
all_embs = []
all_fnames = []
for i, p in enumerate(raws):
    img = extract_thumbnail(p, size=(448, 448))
    crops = detect_player_crops(img, detector)
    if crops:
        embs = embed_crops(crops, embedder, device)
        for row in embs:
            all_embs.append(row)
            all_fnames.append(p.name)
    if (i+1) % 20 == 0:
        print(f'  {i+1}/{len(raws)} ({len(all_embs)} crops so far)')

print(f'\nTotal crops: {len(all_embs)}')
emb_matrix = np.stack(all_embs)

print('\nThreshold → clusters:')
for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    labels = cluster_embeddings(emb_matrix, thresh)
    n = int(labels.max()) + 1
    print(f'  {thresh:.2f}  →  {n} clusters')
