"""
Accuracy test for player coverage clustering on 202605 WA Open MS Raws.
Only uses the labeled range 0001-0696.

Ground truth player assignments (from user):
"""
import re
from pathlib import Path
import numpy as np
import torch

# Build ground truth: file_number -> player_id
# Using distinct IDs for non-same-player groups
RANGES = [
    # (start, end_inclusive, player_id)
    (1,    78,   1),
    (79,   107,  2),
    (108,  126,  3),
    (127,  150,  2),   # same as 79-107
    (151,  177,  4),
    (178,  186,  5),
    (187,  194,  6),
    (195,  227,  5),   # same as 178-186
    (228,  252,  6),   # same as 187-194
    (253,  290,  7),
    (291,  296,  8),
    (297,  327,  9),
    (328,  347,  10),
    (348,  401,  11),
    (402,  442,  12),
    (443,  473,  13),
    (474,  485,  14),
    (486,  509,  15),
    (510,  527,  14),  # same as 474-485
    (528,  575,  16),
    (576,  595,  17),
    (596,  634,  18),
    (635,  653,  19),
    (654,  696,  20),
]

def file_num(fname):
    m = re.search(r'(\d{4,})', Path(fname).stem)
    return int(m.group(1)) if m else -1

def true_player(fname):
    n = file_num(fname)
    for s, e, pid in RANGES:
        if s <= n <= e:
            return pid
    return 0  # unlabeled

raw_dir = Path(r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202605 WA Open MS Raws')
_seen = set()
labeled_raws = []
for p in sorted(raw_dir.glob('*'), key=lambda x: x.name):
    if p.suffix.lower() != '.cr3' or p.name.lower() in _seen:
        continue
    _seen.add(p.name.lower())
    n = file_num(p.name)
    if 1 <= n <= 696:
        labeled_raws.append(p)

print(f'Labeled photos: {len(labeled_raws)}')
n_true_players = len({true_player(p.name) for p in labeled_raws} - {0})
print(f'True players:   {n_true_players}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from inference.player_coverage import build_detector, build_embedder, detect_player_crops, embed_crops, cluster_embeddings
from data.raw_reader import extract_thumbnail

detector = build_detector()
embedder = build_embedder(device)

print('\nDetecting + embedding...')
# Photo-level: one embedding per photo = largest-person crop only
photo_embs = []   # one per photo that had detections
photo_names = []
for i, p in enumerate(labeled_raws):
    img = extract_thumbnail(p, size=(448, 448))
    results = detector(img, classes=[0], verbose=False)
    w, h = img.size
    min_area = w * h * 0.01
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1,y1,x2,y2 = map(int,box)
            if (x2-x1)*(y2-y1) >= min_area:
                boxes.append((x1,y1,x2,y2))
    if not boxes:
        continue
    # Largest box only
    largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    crop = img.crop(largest)
    emb = embed_crops([crop], embedder, device)[0]
    photo_embs.append(emb)
    photo_names.append(p.name)
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(labeled_raws)} ({len(photo_embs)} detected)')

print(f'\nPhotos with detections: {len(photo_embs)}/{len(labeled_raws)}')
emb_matrix = np.stack(photo_embs)

print('\n--- Baseline (one crop per photo, largest person) ---')
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    labels = cluster_embeddings(emb_matrix, thresh)
    n_clusters = int(labels.max()) + 1

    # Check coverage: does each true player have at least one cluster containing them?
    player_to_clusters: dict[int,set] = {}
    cluster_to_players: dict[int,set] = {}
    for i, fname in enumerate(photo_names):
        pid = true_player(fname)
        cid = int(labels[i])
        player_to_clusters.setdefault(pid, set()).add(cid)
        cluster_to_players.setdefault(cid, set()).add(pid)

    # Purity: fraction of clusters that are "pure" (contain only 1 true player)
    pure = sum(1 for ps in cluster_to_players.values() if len(ps) == 1)
    # Coverage: each true player has ≥1 cluster
    covered = set(player_to_clusters.keys()) - {0}

    print(f'thresh={thresh:.2f}: {n_clusters:3d} clusters  pure={pure}/{n_clusters}  '
          f'covered={len(covered)}/{n_true_players}  '
          f'{"OK" if len(covered)==n_true_players else "MISS"}')

print('\n--- Temporal: average embeddings within window of K consecutive photos ---')
for K in [3, 5, 7]:
    # Sliding window average: for each photo, average embeddings of ±K/2 neighbors
    smoothed = []
    for i in range(len(photo_embs)):
        lo = max(0, i - K//2)
        hi = min(len(photo_embs), i + K//2 + 1)
        window = np.stack(photo_embs[lo:hi])
        avg = window.mean(axis=0)
        avg /= np.linalg.norm(avg) + 1e-8
        smoothed.append(avg)
    sm_matrix = np.stack(smoothed)

    best_covered = 0
    best_thresh = None
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        labels = cluster_embeddings(sm_matrix, thresh)
        n_clusters = int(labels.max()) + 1
        player_to_clusters = {}
        for i, fname in enumerate(photo_names):
            pid = true_player(fname)
            player_to_clusters.setdefault(pid, set()).add(int(labels[i]))
        covered = set(player_to_clusters.keys()) - {0}
        if len(covered) > best_covered:
            best_covered = len(covered)
            best_thresh = thresh
        print(f'  K={K} thresh={thresh:.2f}: {n_clusters:3d} clusters  '
              f'covered={len(covered)}/{n_true_players}  '
              f'{"OK" if len(covered)==n_true_players else ""}')
    print()
